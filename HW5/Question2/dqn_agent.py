from network import DeepNetwork
import torch
import torch.optim as optim
import torch.nn as nn
from experience_replay import ReplayBuffer
import numpy as np


class Agent(object):
    # TODO: Define the DQN Agent here.
    #
    # Hints: 1- Initialize a Neural Network (using DeepNetwork class),
    #           and other essential parameters in the init method.
    #        2- Define a method for taking action using 
    #           epsilon-greedy policy based on the DQN outputs.
    #        3- Define a learning method for updating DQN weights.
    #        4- You may also need to define some extra methods
    #           for decrementing epsilon, saving, and loading a trained network.

    def __init__(self, state_dim, action_dim, env):
        self.online_net = DeepNetwork(state_dim, action_dim)
        self.target_net = DeepNetwork(state_dim, action_dim)
        for p in self.target_net.network.parameters():
            p.requires_grad = False
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=0.001)
        self.discount_factor = 0.99
        self.loss_function = nn.MSELoss()
        self.replay_buffer = ReplayBuffer()
        self.env = env

    def select_action(self, state, epsilon):
        if np.random.uniform(0, 1) < epsilon:
                return self.env.action_space.sample()  # choose random action
        else:
                action_q_vals = self.online_net(state).data.numpy()
                return np.argmax(action_q_vals)
        
    
    def update_network(self, state, next_state, action, reward, done):
        state_q_val = torch.gather(self.online_net(state), dim=1, index=action.long())
        next_action_q_val = self.target_net(next_state)
        next_action_q_val, _ = torch.max(next_action_q_val, dim=1, keepdim=True)
        state_continues = 1 - done
        next_state_target = reward + state_continues * self.discount_factor * next_action_q_val
        loss = self.loss_function(state_q_val, next_state_target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_q_values(self):

        
        states, next_states, actions, rewards, done = self.replay_buffer.sample_batch_data(32)
        self.update_network(torch.Tensor(states), torch.Tensor(next_states), torch.Tensor(actions), torch.Tensor(rewards), torch.Tensor(done))
        # self.update_network(states, next_states, actions, torch.Tensor(rewards, torch.Tensor(done))