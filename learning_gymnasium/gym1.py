import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

env = gym.make("CartPole-v1", render_mode="human")

print(f"Action space: {env.action_space}")
print(f"Sample action: {env.action_space.sample()}")

print(f"Observation space: {env.observation_space}")
print(f"Sample observation: {env.observation_space.sample()}")

observation, info = env.reset()

print(f"Starting observation: {observation}")

episode_over = False
total_reward = 0

while not episode_over:
    action = env.action_space.sample() # randoma action

    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()
