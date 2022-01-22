import numpy as np

a = np.random.randint(low=0, high=9, size=(2,9,9))
states = np.array([[i,j] for i in range(2) for j in range(4)])
b = np.max(a, axis=0)
c = np.array(states[2])
print(states)
print(b)
print(c)
print(b[c[0]][c[1]])
