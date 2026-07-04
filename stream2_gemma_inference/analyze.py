"""analyze_run — Stream 2's single deliverable. Composes images/prompts/client/
schema; owns bookkeeping stamping and the one text-only repair round."""

import os
import tempfile
from datetime import datetime, timezone

from .client import GemmaClient
from .errors import AnalyzeRunError
from .images import frame_labels, make_contact_sheet, select_frames
from .prompts import build_messages, build_repair_messages
from .schema import extract_json, validate

DEFAULT_N_FRAMES = 8


def _resolve(value, env_key, default, cast):
    if value is not None:
        return value
    return cast(os.environ.get(env_key, default))


def _frame_items(frames, manifest, n_frames, contact_sheet):
    chosen = select_frames(frames, manifest, n_frames)
    labels = frame_labels(chosen, manifest)
    if not contact_sheet:
        return list(zip(labels, chosen))
    sheet = os.path.join(tempfile.mkdtemp(prefix="stream2_"), "contact_sheet.png")
    make_contact_sheet(chosen, labels, sheet)
    return [(f"CONTACT SHEET of {len(chosen)} rollout frames "
             "(per-tile labels inside the image)", sheet)]


def _parse_or_problems(raw, manifest):
    try:
        candidate = extract_json(raw)
    except ValueError as exc:
        return None, [str(exc)]
    problems = validate(candidate, manifest)
    return (candidate, problems)


def analyze_run(frames, chart, hypothesis_log, manifest=None, *,
                client=None, n_frames=None, contact_sheet=None):
    """Analyze one training run's evidence; return the updated hypothesis entry
    (root README data-contract shape, cumulative state — persist verbatim)."""
    client = client or GemmaClient()
    n_frames = _resolve(n_frames, "GEMMA_N_FRAMES", DEFAULT_N_FRAMES, int)
    contact_sheet = _resolve(contact_sheet, "GEMMA_CONTACT_SHEET", "0",
                             lambda v: str(v) not in ("0", "false", "False", ""))

    items = _frame_items(frames, manifest, n_frames, contact_sheet)
    raw = client.chat(build_messages(items, chart, hypothesis_log, manifest))
    candidate, problems = _parse_or_problems(raw, manifest)

    if problems:  # one text-only repair round at temperature 0
        raw = client.chat(build_repair_messages(problems, raw), temperature=0)
        candidate, problems = _parse_or_problems(raw, manifest)
        if problems:
            raise AnalyzeRunError(
                "bad_json", f"model output failed validation after repair: {problems}",
                raw_response=raw)

    # Bookkeeping is ours, never the model's.
    run = manifest["run"] if manifest else hypothesis_log.get("run")
    candidate["run"] = run
    candidate["timestamp"] = datetime.now(timezone.utc).isoformat()
    if manifest is None:
        # Without telemetry the model has no training-step vocabulary; any
        # range it emitted would send Stream 4's browser somewhere meaningless.
        candidate["next_to_check"] = None
    elif candidate.get("next_to_check"):
        candidate["next_to_check"]["run"] = run
    return candidate
