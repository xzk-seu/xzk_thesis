import torch
import numpy as np
t = torch.arange(0, 16).reshape(4, -1)
print(t)

# t1 = t[[1, 3, 1], [0, 1, 2]]
# print(t1)
# print(t.shape, t1.shape)

mask = t > 7
print(mask)
print(mask.dtype)

t1 = t[mask]
print(t1)
print(t1.dtype)

# t1 = torch.index_select(t, 1, indices)
# print(t1)
#
# t1 = torch.gather(t, 0, indices)
# print(t1)
