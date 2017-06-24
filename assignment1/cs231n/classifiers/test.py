import numpy as np
from past.builtins import xrange

x = np.array([[1, 2, 3], [4, 5, 6]])


a = np.bincount(np.array([0, 1, 2, 2, 2, 2, 3]))
b = np.bincount(np.array([0, 1, 1, 3, 2, 1, 7]))
c = np.argmax(np.bincount(np.array([0, 1, 1, 3, 2, 1, 7])))


d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

test = np.maximum(0, [1, 2, 3, -1])

number = d[d > 1]

A = np.arange(10)
split = np.split(A, 5)
split = np.array(split)
print(split)
print(split[0])
print(split[[0,1,2,3]])

# for i in range(5):
#     test_data = split[i]
#     print(np.delete(np.arange(5), i))
#     train_data = split[np.delete(np.arange(5), i)]
#     train_data = np.concatenate(train_data)
    
#     # print("Train:\n{}".format(train_data))
#     # print("Test:\n{}".format(test_data))
#     print()