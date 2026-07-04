"""Image plumbing: base64 data-URIs, event-frame-pinned subsampling, sim-step
labels (by filename basename, never list index), and the contact-sheet latency
lever (per-image visual-token cost is fixed regardless of pixel size, so
tiling N frames into one image is the real token cut)."""

import base64
import math
import os

from PIL import Image, ImageDraw

LABEL_STRIP_PX = 22


def encode_image(path):
    with open(path, "rb") as f:
        payload = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _basename_to_sim_step(manifest):
    if not manifest:
        return {}
    return {
        os.path.basename(rec["file"]): rec["sim_step"] for rec in manifest.get("frames", [])
    }


def select_frames(frames, manifest, n_frames):
    """Subsample to n_frames, always keeping first, last, and the manifest's
    drop_step frame — uniform-with-endpoints subsampling provably deletes the
    single most evidence-bearing frame (index 13 of 15 in all three fixtures)."""
    n_frames = max(3, n_frames)
    if len(frames) <= n_frames:
        return list(frames)

    pinned = {0, len(frames) - 1}
    step_map = _basename_to_sim_step(manifest)
    if manifest and "drop_step" in manifest:
        for i, path in enumerate(frames):
            if step_map.get(os.path.basename(path)) == manifest["drop_step"]:
                pinned.add(i)
                break

    remaining = n_frames - len(pinned)
    candidates = [i for i in range(len(frames)) if i not in pinned]
    if remaining > 0 and candidates:
        stride = len(candidates) / remaining
        pinned.update(
            candidates[min(int(k * stride), len(candidates) - 1)] for k in range(remaining)
        )
    return [frames[i] for i in sorted(pinned)]


def frame_labels(frames, manifest):
    step_map = _basename_to_sim_step(manifest)
    labels = []
    for i, path in enumerate(frames):
        step = step_map.get(os.path.basename(path))
        labels.append(f"frame {i} = sim step {step}" if step is not None else f"frame {i}")
    return labels


def make_contact_sheet(frames, labels, out_path, cols=4):
    tiles = [Image.open(p).convert("RGB") for p in frames]
    w, h = tiles[0].size
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * w, rows * (h + LABEL_STRIP_PX)), "black")
    draw = ImageDraw.Draw(sheet)
    for i, (tile, label) in enumerate(zip(tiles, labels)):
        x = (i % cols) * w
        y = (i // cols) * (h + LABEL_STRIP_PX)
        sheet.paste(tile, (x, y))
        draw.text((x + 4, y + h + 4), label, fill="white")
    sheet.save(out_path)
    return out_path
