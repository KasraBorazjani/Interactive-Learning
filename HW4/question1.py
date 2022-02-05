from model import NSFrozenLake
import numpy as np
from amalearn.agent import AgentBase

STUDENT_NUM = 810196662
env = NSFrozenLake(STUDENT_NUM)

class valueIterationAgent(AgentBase):

    def __init__(self, env, id, discount, theta):
        super(valueIterationAgent, self).__init__(id, env)
        self.gamma = discount
        self.theta = theta
        self.values = np.zeros((4,4))
        self.q_table = np.zeros((4,4,4))
        self.policy = np.random.randint(0, 4, size=(4,4))
    
    def calc_q_value(self, action, state_now):
        final_reward = 0
        next_states, probs, fail_probs, dones = self.environment.possible_consequences(action, state_now)
        for i in range(len(next_states)):
            reward = 
            final_reward += probs[i] * (rew)

        final_reward += prob * (reward + self.discount * self.values[s_prime[0]][s_prime[1]])
        return final_reward

    def pi_qvalues(self, state):
        q_values_s = np.array([])
        for action in self.actions:
            s_prime_reward = self.calc_q_value(state, action)
            q_values_s = np.append(q_values_s, s_prime_reward)
        return q_values_s 
    
    def value_iteration(self):
        while(True):
            delta = 0
            for s in self.environment.states:
                temp_value_s = self.values[s[0]][s[1]]
                q_values_s = self.pi_qvalues(s)
                self.values[s[0]][s[1]] = max(q_values_s)
                delta = max(delta, abs(temp_value_s - self.values[s[0]][s[1]]))
            
            if delta < self.theta:
                break

        for s in self.environment.states:
            self.policy[s[0]][s[1]] = np.argmax(self.pi_qvalues(s)) 
        
        return self.policy


q_table = np.zeros((4,4,4))
policy = 

state = env.reset()
print(state)

states, probs, fail_probs, dones = env.possible_consequences(1, state)

print("states: ", states)
print("probs: ", probs)
print("fail_probs: ", fail_probs)
print("dones: ", dones)
