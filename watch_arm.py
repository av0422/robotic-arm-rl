import gymnasium as gym
from stable_baselines3 import SAC

env = gym.make("Reacher-v5", render_mode="human")
model = SAC.load("models/reacher_trained")

obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
