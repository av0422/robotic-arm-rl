import gymnasium as gym
from stable_baselines3 import SAC
from gymnasium.wrappers import RecordVideo

# ── Load best model ────────────────────────────────────
model = SAC.load("models/best_model")

# ── Wrap env with video recorder ───────────────────────
env = RecordVideo(
    gym.make("Reacher-v5", render_mode="rgb_array"),
    video_folder="./videos/",
    name_prefix="reacher_trained"
)

obs, info = env.reset()
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
print("Video saved to ./videos/ ✅")
