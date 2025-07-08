import gymnasium as gym
import torch
from dqn_pendulum_agent import DQNPendulumAgent

env = gym.make("CartPole-v1", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_net = DQNPendulumAgent(state_dim, action_dim)
q_net.load_state_dict(torch.load("dqn_agent.pth"))
q_net.eval()

state, _ = env.reset()
print(f"Starting observation: {state}")
done = False
total_reward = 0

step = 0
max_steps = 200
while not done and step < max_steps:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = q_net(state_tensor)
    action = int(torch.argmax(q_values, dim=1))
    state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    done = terminated or truncated

    step += 1
    
print(f"Episode finished! Total reward: {total_reward}")
env.close()
