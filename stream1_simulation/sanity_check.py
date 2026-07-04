"""Quick smoke test: random policy, confirms env physics and headless camera
capture work before spending time on real training. Should finish in <30s.
"""

import os

import numpy as np
from PIL import Image

from env import PushEnv

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "_sanity_check")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    env = PushEnv()
    rng = np.random.default_rng(0)

    saw_drop = False
    saw_frame = False
    episode_returns = []

    for episode in range(5):
        obs, _ = env.reset()
        total_reward = 0.0
        for t in range(200):
            action = rng.uniform(-1, 1, size=2).astype(np.float32)
            # Bias the action toward pushing the object forward (+y) so a
            # random policy actually demonstrates the push/drop behavior
            # instead of just wandering.
            action[1] = abs(action[1])
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if t == 5 and episode == 0:
                frame = env.get_camera_image()
                assert frame.shape == (320, 320, 3), f"unexpected frame shape {frame.shape}"
                Image.fromarray(frame).save(os.path.join(OUT_DIR, "sample_frame.png"))
                saw_frame = True

            if info.get("dropped"):
                saw_drop = True
            if terminated or truncated:
                break
        episode_returns.append(total_reward)
        print(f"episode {episode}: return={total_reward:.2f} dropped={info.get('dropped')} success={info.get('success')}")

    env.close()

    assert saw_frame, "camera capture never ran"
    print(f"\nsaved sample frame to {OUT_DIR}/sample_frame.png")
    print(f"observed at least one drop across 5 random episodes: {saw_drop}")
    print("SANITY CHECK PASSED")


if __name__ == "__main__":
    main()
