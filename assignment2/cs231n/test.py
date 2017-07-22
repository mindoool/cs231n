import numpy as np

a = np.array([[[1,-2,3], [-4,5,-6], [7, 8, 9]], [[1,-2,3], [-4,5,-6], [7, 8, 9]]])
b = np.maximum(a, 0)
print(a.shape)
print(a.shape[1:])
print(b)