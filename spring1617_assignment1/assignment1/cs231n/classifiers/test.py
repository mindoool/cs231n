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

print(x[1,1])
print(x[1,2])

d = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(np.reshape(d, (3,-1)))

test = np.maximum(0, [1,2,3,-1])
print(test)

number = d[d>1] 

x = np.arange(9.0)
np.split(x, 3)
