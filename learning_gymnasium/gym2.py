import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

env = gym.make("CarRacing-v3")
env.observation_space.shape

wrapped_env = FlattenObservation(env)
wrapped_env.observation_space.shape