from collections import defaultdict
from collections import deque
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

from dqn_pendulum_agent import DQNPendulumAgent

def select_action(state, q_net, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = q_net(state_tensor)
        return int(torch.argmax(q_values, dim=1))
    
# Agent traning

def train():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = DQNPendulumAgent(state_dim, action_dim)
    target_net = DQNPendulumAgent(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    replay_buffer = deque(maxlen=10000)
    batch_size = 64
    gamma = 0.99 # discount rate

    epsilon = 1.0 # exploration at the beggining at maximum
    epsilon_decay = 0.995
    min_epsilon = 0.05

    target_update_freq = 10
    num_episodes = 500

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        for t in range(200):
            action = select_action(state, q_net, epsilon, action_dim)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(replay_buffer) >= batch_size:
                minibatch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*minibatch)

                states_tensor = torch.FloatTensor(states)
                actions_tensor = torch.LongTensor(actions).unsqueeze(1)
                rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
                next_states_tensor = torch.FloatTensor(next_states)
                dones_tensor = torch.FloatTensor(dones).unsqueeze(1)

                current_q = q_net(states_tensor).gather(1, actions_tensor)
                next_q = target_net(next_states_tensor).max(1)[0].unsqueeze(1)
                expected_q = rewards_tensor + (1 - dones_tensor) * gamma * next_q

                loss = loss_fn(current_q, expected_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        if episode % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    torch.save(q_net.state_dict(), "dqn_agent.pth")
    print("Model zapisany jako dqn_agent.pth")

    env.close()

if __name__ == "__main__":
    train()


