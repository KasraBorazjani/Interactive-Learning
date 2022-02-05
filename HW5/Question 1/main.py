import numpy as np
import gym
from sarsaAgent import SarsaAgent
from mountainCarTileCoder import MountainCarTileCoder
import matplotlib.pyplot as plt

MAX_TIME = 15000
INIT_EPS = 1
D_FACTOR = 0.95
IHT_SIZE = 4096
POS_MAX = 1.2
POS_MIN = -0.5
V_MAX = 0.7
V_MIN = -0.7
EPSILON_DECAY = 0.95
LEARNING_RATE = 0.5
EPISODES = 200
num_tiles = [16, 4, 8]
num_tilings = [2, 32, 8]
actions = [0, 1, 2]


gym.envs.register(
    id='MountainCarVersion-v1',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=MAX_TIME,
    reward_threshold=-1,     
)
env = gym.make('MountainCarVersion-v1')

agent0 = SarsaAgent(env, INIT_EPS, D_FACTOR, IHT_SIZE, num_tiles[0], num_tilings[0], actions, POS_MAX, POS_MIN, V_MAX, V_MIN, 0.995, 0.2, MAX_TIME)
agent1 = SarsaAgent(env, INIT_EPS, D_FACTOR, IHT_SIZE, num_tiles[1], num_tilings[1], actions, POS_MAX, POS_MIN, V_MAX, V_MIN, 0.9, 0.6, MAX_TIME)
agent2 = SarsaAgent(env, INIT_EPS, D_FACTOR, IHT_SIZE, num_tiles[2], num_tilings[2], actions, POS_MAX, POS_MIN, V_MAX, V_MIN, 0.95, 0.5, MAX_TIME)
step_per_episode_0 = agent0.learn(EPISODES)
step_per_episode_1 = agent1.learn(EPISODES)
step_per_episode_2 = agent2.learn(EPISODES)

plt.plot(range(len(step_per_episode_0)), [step_per_episode_0, step_per_episode_1, step_per_episode_2])
plt.legend(["configuration 1", "configuration 2", "configuration 3"])
plt.xlabel("episodes")
plt.ylabel("steps")
plt.savefig("HW5/Question 1/output")




