"""
Generates placeholder rollout frames + a reward curve PNG so Stream 4 can
build/test the Gradio UI before Stream 1's real sim outputs exist.

Run: python generate_placeholders.py
"""
import os
import random

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(HERE, "sample_frames")
os.makedirs(FRAMES_DIR, exist_ok=True)


def make_frame(i: int, n: int) -> None:
    img = Image.new("RGB", (320, 240), color=(30, 30, 40))
    draw = ImageDraw.Draw(img)
    # fake "object" sliding across the frame, dropping near the end
    x = int(20 + (240 * i / n))
    y = 120 if i < n * 0.8 else 120 + int((i - n * 0.8) * 6)
    draw.ellipse([x, y, x + 30, y + 30], fill=(200, 80, 80))
    draw.rectangle([0, 200, 320, 240], fill=(60, 60, 60))  # table
    draw.text((10, 10), f"frame {i:03d}", fill=(255, 255, 255))
    img.save(os.path.join(FRAMES_DIR, f"frame_{i:03d}.png"))


def make_reward_curve() -> None:
    img = Image.new("RGB", (500, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.line([(40, 260), (40, 20)], fill=(0, 0, 0), width=2)  # y axis
    draw.line([(40, 260), (480, 260)], fill=(0, 0, 0), width=2)  # x axis

    points = []
    reward = 0.0
    for step in range(0, 60000, 1000):
        reward += random.uniform(-0.5, 1.0)
        if 47000 <= step <= 52000:
            reward -= 2.0  # the flagged dip
        x = 40 + int(step / 60000 * 420)
        y = 260 - int(max(0, min(200, reward * 3)))
        points.append((x, y))
    draw.line(points, fill=(200, 40, 40), width=2)

    # highlight flagged region
    x0 = 40 + int(47000 / 60000 * 420)
    x1 = 40 + int(52000 / 60000 * 420)
    draw.rectangle([x0, 20, x1, 260], outline=(0, 120, 255), width=2)
    draw.text((x0, 5), "flagged", fill=(0, 120, 255))

    img.save(os.path.join(HERE, "reward_curve.png"))


if __name__ == "__main__":
    n_frames = 40
    for i in range(n_frames):
        make_frame(i, n_frames)
    make_reward_curve()
    print(f"Wrote {n_frames} frames to {FRAMES_DIR}")
    print(f"Wrote reward_curve.png to {HERE}")
