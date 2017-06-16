import numpy as np
from past.builtins import xrange

x = np.array([[1, 2, 3], [4, 5, 6]])
print(x[1, ])
print(x[1, :])

a = np.bincount(np.array([0, 1, 2, 2, 2, 2, 3]))
b = np.bincount(np.array([0, 1, 1, 3, 2, 1, 7]))
c = np.argmax(np.bincount(np.array([0, 1, 1, 3, 2, 1, 7])))
print(a)
print(b)
print(c)