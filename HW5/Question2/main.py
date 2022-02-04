from time import strftime
import numpy as np
from pendulum import time
from dqn_agent import Agent
import gym
import torch
import matplotlib.pyplot as plt
import datetime


env = gym.make('LunarLander-v2')

action_dim = env.action_space.n
state_dim = env.observation_space.shape[0]

agent = Agent(state_dim, action_dim, env)
EPISODES = 6000
MAX_STEPS = 2000
epsilon = 1
SYNC_EVERY = 1000

rewards = []

## copied from external sources
def mean_rewards_over_window(values, window=50):
    weight = np.repeat(1.0, window)/window
    means_list = np.convolve(values,weight,'valid')
    return means_list

def plot_rewards(mean_rewards):
    plt.plot(range(len(mean_rewards)), mean_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Mean Rewards')
    plt.savefig("HW5/Question2/checkpoints/"+datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

steps = 0
for episode in range(EPISODES):
    reward_sum = 0
    state = env.reset()

    for _ in range(MAX_STEPS):
        
        if (steps % SYNC_EVERY == 0):
            agent.target_net.network.load_state_dict(agent.online_net.network.state_dict())
        
        action = agent.select_action(torch.from_numpy(state).float(), epsilon)
        next_state, reward, terminal, _ = env.step(action)
        
        reward_sum += reward
        agent.replay_buffer.add_sample((state,next_state,[action],[reward],[terminal]))
        state = next_state
        steps += 1

        if terminal:
            rewards.append(reward_sum)
            if episode % 200 == 0:
                print('episode:', episode, 'sum_of_rewards_for_episode:', reward_sum)
                mean_rewards = mean_rewards_over_window(rewards)
                plot_rewards(mean_rewards)
            break
            
    agent.update_q_values()
    
    epsilon = max(epsilon*0.99, 0.2)



