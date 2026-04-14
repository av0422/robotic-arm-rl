import gymnasium as gym

# This is a proper robotic arm - Reacher!
env = gym.make("Reacher-v5", render_mode="human")
obs, info = env.reset()

print("Robotic Arm loaded! ✅")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Watch it move randomly
for _ in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
print("Done!")
