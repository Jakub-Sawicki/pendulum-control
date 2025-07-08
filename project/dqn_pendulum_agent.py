import torch.nn as nn
import torch.optim as optim

class DQNPendulumAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNPendulumAgent, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)
