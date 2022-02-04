import numpy as np
import tiles3 as tc
from mountainCarTileCoder import MountainCarTileCoder

class SarsaAgent():
    def __init__(self, init_eps, d_factor, iht, num_tiles, num_tilings, actions_len, pos_max, pos_min, v_max, v_min):
        self.epsilon = init_eps
        self.gamma = d_factor
        self.iht_size = iht
        self.num_tiles = num_tiles
        self.num_tilings = num_tilings
        self.action_space_len = actions_len
        self.tileCoder = MountainCarTileCoder(self.iht_size, self.num_tilings, self.num_tiles, pos_max, pos_min, v_max, v_min)
        self.q_table = np.zeros((self.num_tilings, self.iht_size))
        

