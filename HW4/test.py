import numpy as np

a = np.array([[1,2,3], [4,5,6]])
states = [(0,2), (1,1)]
states_T = np.transpose(np.array(states))
print(a[states_T])