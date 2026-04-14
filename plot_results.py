import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3 import SAC
import gymnasium as gym

# ── 1. Load training logs ──────────────────────────────
log_path = "./logs/evaluations.npz"
data = np.load(log_path)

timesteps = data['timesteps']
results = data['results']
mean_rewards = results.mean(axis=1)
std_rewards = results.std(axis=1)

# ── 2. Plot ────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.plot(timesteps, mean_rewards, color='royalblue', linewidth=2, label='Mean Reward')
plt.fill_between(timesteps,
                 mean_rewards - std_rewards,
                 mean_rewards + std_rewards,
                 alpha=0.2, color='royalblue', label='Std Dev')

plt.xlabel('Timesteps', fontsize=12)
plt.ylabel('Mean Reward', fontsize=12)
plt.title('Robotic Arm Training Progress (SAC on Reacher-v5)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_curve.png', dpi=150)
plt.show()
print("Plot saved as training_curve.png")
