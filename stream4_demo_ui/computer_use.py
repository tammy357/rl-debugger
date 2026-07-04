"""
Stream 4 — Computer Use integration.

Given a run + timestep range, drive a headless browser to the matching
WandB chart view and return a screenshot. Falls back to a mock screenshot
if WANDB_RUN_URL isn't set, so this can be developed before Stream 1's
real logging exists.

Contract (from README):
  input:  {"run": 2, "step_range": [47000, 52000]}
  output: path to a screenshot (str)
"""
import os
from dataclasses import dataclass

SCREENSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


@dataclass
class TimestepRequest:
    run: int
    step_range: tuple[int, int]


def fetch_wandb_screenshot(request: TimestepRequest, wandb_run_url: str | None = None) -> str:
    """
    Navigate to the WandB chart for `request.run`, scrub/zoom to
    `request.step_range`, and screenshot it.

    Returns the local path to the saved screenshot.
    """
    wandb_run_url = wandb_run_url or os.environ.get("WANDB_RUN_URL")

    if not wandb_run_url:
        # Mock mode — no real WandB run configured yet.
        return _mock_screenshot(request)

    from playwright.sync_api import sync_playwright

    out_path = os.path.join(
        SCREENSHOT_DIR, f"run{request.run}_{request.step_range[0]}_{request.step_range[1]}.png"
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(wandb_run_url, wait_until="networkidle")

        # TODO: once Stream 1's real WandB workspace exists, fill in the
        # actual selectors for zooming the reward-curve chart to
        # request.step_range. WandB charts are usually SVG/canvas — the
        # simplest robust approach is often the WandB API/export rather
        # than pixel-perfect UI scrubbing, if a screenshot isn't a hard
        # requirement for the demo.
        page.wait_for_timeout(1500)
        page.screenshot(path=out_path)
        browser.close()

    return out_path


def _mock_screenshot(request: TimestepRequest) -> str:
    """Placeholder screenshot so the UI can be built before real WandB exists."""
    from PIL import Image, ImageDraw

    out_path = os.path.join(
        SCREENSHOT_DIR, f"mock_run{request.run}_{request.step_range[0]}_{request.step_range[1]}.png"
    )
    img = Image.new("RGB", (640, 400), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), f"[MOCK] WandB view — Run {request.run}", fill=(0, 0, 0))
    draw.text((20, 50), f"Steps {request.step_range[0]}-{request.step_range[1]}", fill=(0, 0, 0))
    draw.rectangle([20, 90, 620, 380], outline=(0, 120, 255), width=2)
    img.save(out_path)
    return out_path


if __name__ == "__main__":
    req = TimestepRequest(run=2, step_range=(47000, 52000))
    path = fetch_wandb_screenshot(req)
    print(f"Screenshot saved to: {path}")
