import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

# Load trained model
env = gym.make("Reacher-v5")
model = SAC.load("models/best_model")

# Evaluate over 20 episodes
mean_reward, std_reward = evaluate_policy(
    model, env,
    n_eval_episodes=20,
    deterministic=True
)

print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"Mean Reward : {mean_reward:.2f}")
print(f"Std Dev     : {std_reward:.2f}")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(f"✅ Model evaluated over 20 episodes")
