import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

# ── Load existing trained model and keep training ──────
env = gym.make("Reacher-v5")
eval_env = gym.make("Reacher-v5")

# Load your already trained model
model = SAC.load("models/reacher_trained", env=env)

print("Continuing training from -3.53 reward...")
print("Target: push toward 0!")
print("─" * 50)

# Eval callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/",
    log_path="./logs/",
    eval_freq=5000,
    verbose=1
)

# Train 100k MORE steps
model.learn(
    total_timesteps=100_000,
    callback=eval_callback,
    reset_num_timesteps=False  # continues from step 100k
)

model.save("models/reacher_best")
print("Done! ✅ Saved as reacher_best")
