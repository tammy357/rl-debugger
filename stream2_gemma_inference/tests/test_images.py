import base64

from PIL import Image

from stream2_gemma_inference.images import (
    encode_image,
    frame_labels,
    make_contact_sheet,
    select_frames,
)


def test_encode_image_is_png_data_uri(run1):
    frames, _, _ = run1
    uri = encode_image(frames[0])
    assert uri.startswith("data:image/png;base64,")
    base64.b64decode(uri.split(",", 1)[1])  # decodes cleanly


def test_select_keeps_all_when_budget_covers(run1):
    frames, _, manifest = run1
    assert select_frames(frames, manifest, 15) == frames


def test_select_pins_drop_frame_for_any_budget(run1):
    frames, _, manifest = run1
    drop_file = next(
        rec["file"] for rec in manifest["frames"] if rec["sim_step"] == manifest["drop_step"]
    )
    for n in range(3, 15):
        chosen = select_frames(frames, manifest, n)
        assert len(chosen) == n
        basenames = [f.rsplit("/", 1)[-1] for f in chosen]
        assert drop_file.rsplit("/", 1)[-1] in basenames, f"drop frame lost at n={n}"
        assert chosen[0] == frames[0] and chosen[-1] == frames[-1]
        assert chosen == sorted(chosen, key=frames.index)  # temporal order


def test_select_without_manifest_keeps_endpoints(run1):
    frames, _, _ = run1
    chosen = select_frames(frames, None, 8)
    assert len(chosen) == 8 and chosen[0] == frames[0] and chosen[-1] == frames[-1]


def test_labels_use_sim_steps_from_manifest(run1):
    frames, _, manifest = run1
    labels = frame_labels(frames, manifest)
    assert labels[0] == "frame 0 = sim step 0"
    assert labels[13] == f"frame 13 = sim step {manifest['drop_step']}"


def test_labels_by_basename_not_index(run1):
    frames, _, manifest = run1
    labels = frame_labels(list(reversed(frames)), manifest)
    assert labels[0] == f"frame 0 = sim step {manifest['frames'][-1]['sim_step']}"


def test_labels_plain_without_manifest(run1):
    frames, _, _ = run1
    assert frame_labels(frames[:2], None) == ["frame 0", "frame 1"]


def test_contact_sheet_dimensions(run1, tmp_path):
    frames, _, manifest = run1
    labels = frame_labels(frames, manifest)
    out = make_contact_sheet(frames, labels, str(tmp_path / "sheet.png"), cols=4)
    sheet = Image.open(out)
    assert sheet.width == 4 * 320          # 4 tiles wide
    assert sheet.height == 4 * (320 + 22)  # 15 frames -> 4 rows, 22px label strip
