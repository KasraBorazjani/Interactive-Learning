import numpy as np
import gym
from sarsaAgent import SarsaAgent
from mountainCarTileCoder import MountainCarTileCoder
import matplotlib.pyplot as plt

MAX_TIME = 15000
init_eps = 0.9
d_factor = 0.8
IHT_SIZE = 4096
num_tiles = [16, 4, 8]
num_tilings = [2, 32, 8]
actions = [0, 1, 2]
POS_MAX = 1.2
POS_MIN = -0.5
V_MAX = 0.7
V_MIN = -0.7
EPSILON_DECAY = 0.99
LEARNING_RATE = 0.5
EPISODES = 200


gym.envs.register(
    id='MountainCarVersion-v1',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=MAX_TIME, # MountainCar-v0 uses 200
    reward_threshold=-1,     
)
env = gym.make('MountainCarVersion-v1')

agent0 = SarsaAgent(env, init_eps, d_factor, IHT_SIZE, num_tiles[0], num_tilings[0], actions, POS_MAX, POS_MIN, V_MAX, V_MIN, EPSILON_DECAY, LEARNING_RATE, MAX_TIME)

step_per_episode = agent0.learn(EPISODES)

plt.plot(range(len(step_per_episode)), step_per_episode)
plt.show()






