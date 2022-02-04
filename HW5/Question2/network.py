import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepNetwork(nn.Module):
    

    def __init__(self, state_dim , action_dim):
        super(DeepNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 200),
            nn.ReLU(),
            nn.Linear(200,150),
            nn.ReLU(),
            nn.Linear(150, action_dim)
        )
        

    def forward(self, state):
        
        return self.network(state)