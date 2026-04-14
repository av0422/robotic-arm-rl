import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import os

# ── 1. Create environment ──────────────────────────────
env = gym.make("Reacher-v5")
eval_env = gym.make("Reacher-v5", render_mode="human")

# ── 2. Create SAC model ────────────────────────────────
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,                # prints training progress
    learning_rate=3e-4,
    batch_size=256,
    gamma=0.99,
    tensorboard_log="./logs/"
)

# ── 3. Save best model automatically ──────────────────
os.makedirs("models", exist_ok=True)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/",
    log_path="./logs/",
    eval_freq=5000,
    verbose=1
)

# ── 4. Train! ──────────────────────────────────────────
print("Training started... ⏳")
print("Watch the reward go UP over time!")
print("─" * 50)

model.save("models/reacher_start")   # save untrained model
model.learn(
    total_timesteps=100_000,         # ~5 mins on M1
    callback=eval_callback
)
model.save("models/reacher_trained") # save trained model
print("Training complete! ✅")
