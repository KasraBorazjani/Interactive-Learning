import numpy as np

a = np.array([[1,1], [1,-1], [1,0], [0,1], [0,-1], [0,0], [-1,1], [-1,-1], [-1,0]])

b = np.array([1,1])

print(b in a)