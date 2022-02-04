import queue
import numpy as np
import torch

class ReplayBuffer():

    def __init__(self, max_len=1e5):
        self.buffer = []
        self.max_len = max_len

    def add_sample(self, data):
        if len(self.buffer) < self.max_len:
            self.buffer.append(data)
        else:
            self.buffer.pop(0)
            self.buffer.append(data)
    
    def sample_batch_data(self, length):
        state_list = []
        next_state_list = []
        action_list = []
        reward_list = []
        done_list = []
        # rand_samples = np.random.randint(low = 0, high=len(self.buffer)+1,size=(1,length))
        for i in range(length):
            rand_samples = np.random.randint(0,len(self.buffer)-1)
            buffer_sample = self.buffer[rand_samples]
            # print(self.buffer[rand_samples])
            state_list.append(buffer_sample[0])
            next_state_list.append(buffer_sample[1])
            action_list.append(buffer_sample[2])
            reward_list.append(buffer_sample[3])
            done_list.append(buffer_sample[4])
        
        return torch.Tensor(state_list), torch.Tensor(next_state_list), torch.Tensor(action_list), torch.Tensor(reward_list), torch.Tensor(done_list)
        
