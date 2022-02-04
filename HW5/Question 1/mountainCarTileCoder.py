import numpy as np
import tiles3 as tc

# Tile Coding Function
class MountainCarTileCoder:
    def __init__(self, iht_size, num_tilings, num_tiles, pos_max, pos_min, v_max, v_min):
        """
        Initializes the MountainCar Tile Coder
        *Initializers:
        iht_size(int): the size of the index hash table, typically a power of 2. 4096 in this question.
        num_tilings(int): the number of tilings
        num_tiles(int: the number of tiles. Here both the width and height of the tile coder are the same

        *Class Variables:
        self.iht(tc.IHT): the index hash table that the tile coder will use, better understood after reviewing the link provided in the question
        self.num_tilings(int): the number of tilings the tile coder will use
        self.num_tiles(int): the number of tiles the tile coder will use
        """
        self.iht = tc.IHT(iht_size)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        self.pos_max = pos_max
        self.pos_min = pos_min
        self.v_max = v_max
        self.v_min = v_min 
        self.position_tile_scale = self.num_tiles/(self.pos_max - self.pos_min)
        self.velocity_tile_scale = self.num_tiles/(self.v_max - self.v_min)
    
    def get_tiles(self, position, velocity):
        """
        Takes in a position and velocity from the mountaincar environment
        and returns a numpy array of active tiles.
        
        Arguments:
        position(float): the position of the agent 
        velocity(float): the velocity of the agent 
        returns:
        tiles - np.array, active tiles
        """
        
        # Scale position and velocity by multiplying the inputs of each by their scale
        velocity_scaled = velocity * self.velocity_tile_scale
        position_scaled = position * self.position_tile_scale

        # get the tiles using tc.tiles, with self.iht, self.num_tilings and [scaled position, scaled velocity]
        tiles_list = tc.tiles(self.iht, self.num_tilings, [position_scaled, velocity_scaled])
        
        return np.array(tiles_list)