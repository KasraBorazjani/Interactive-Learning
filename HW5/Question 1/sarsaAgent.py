import numpy as np
from mountainCarTileCoder import MountainCarTileCoder

class SarsaAgent():
    def __init__(self, env, init_eps, d_factor, iht, num_tiles, num_tilings, actions, pos_max, pos_min, v_max, v_min, epsilon_decay, learning_rate, max_steps):
        self.epsilon = init_eps
        self.gamma = d_factor
        self.iht_size = iht
        self.num_tiles = num_tiles
        self.num_tilings = num_tilings
        self.actions = actions
        self.action_space_len = len(self.actions)
        self.epsilon_decay = epsilon_decay
        self.tile_coder = MountainCarTileCoder(self.iht_size, self.num_tilings, self.num_tiles, pos_max, pos_min, v_max, v_min)
        self.q_table = np.ones((self.action_space_len, self.iht_size))
        self.env = env
        self.lr = learning_rate
        self.max_steps = max_steps
        self.step_per_episode = []

    def select_action(self, tiles):
        
        if(np.random.uniform(0,1)<self.epsilon):
            return np.random.randint(low=0, high=self.action_space_len)
        
        else:
            action_values = []
            for i in range(self.action_space_len):
                action_values.append(np.sum(self.q_table[i, tiles]))
            
            return np.argmax(action_values)
    
    def step_learn (self, state, action_index):
        position, velocity = state
        # print("Position: ", position, " velocity: ", velocity)
        tiles_to_change = self.tile_coder.get_tiles(position, velocity)
        # print(tiles_to_change)
        next_state, reward, done, info = self.env.step(self.actions[action_index])
        if done:
            for tile in tiles_to_change:
                self.q_table[action_index, tile] += self.lr*(reward - self.q_table[action_index, tile])
            return next_state, None, reward, done

        next_position, next_velocity = next_state
        next_tiles_to_change = self.tile_coder.get_tiles(next_position, next_velocity)
        next_action_index = self.select_action(next_tiles_to_change)
        next_action = self.actions[next_action_index]
        # next_value = np.sum(self.q_table[next_action, next_tiles_to_change])
        
        for tile in tiles_to_change:
            next_value = self.q_table[next_action_index, tile]
            self.q_table[action_index, tile] += self.lr*(reward + self.gamma*next_value - self.q_table[action_index, tile])
            # print("update to q_value")
            # print(reward - self.q_table[action_index, tile])
        
        return next_state, next_action_index, reward, done
    
    def learn(self, episodes):
        
        for episode in range(episodes):
            state = self.env.reset()
            position, velocity = state
            # print("Position: ", position, " velocity: ", velocity)
            tiles_to_change = self.tile_coder.get_tiles(position, velocity)
            # print(tiles_to_change)
            action_index = self.select_action(tiles_to_change)
            steps = 0
            self.epsilon = max(self.epsilon*self.epsilon_decay, 0.1)
            self.lr = max(self.lr*0.8, 1e-6)
            
            while(True):
                # print(steps)

                next_state, next_action_index, reward, done= self.step_learn(state, action_index)

                if done:
                    self.step_per_episode.append(steps)
                    print("Episode: {}, epsilon: {}, steps: {}".format(episode, self.epsilon, steps))
                    break

                state = next_state
                action_index = next_action_index
                steps += 1
        
            # print(self.q_table)
            
        return self.step_per_episode
                








        
        







        



