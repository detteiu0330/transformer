import numpy as np


inf = -float('inf')
a = np.array([[1, 1, 1], [inf, inf, inf], [inf, inf, inf]])
b = np.array([[1, 1, 1], [1, 1, 1], [2, 2, 2]])
c = np.matmul(a, np.transpose(b))
print(c)
