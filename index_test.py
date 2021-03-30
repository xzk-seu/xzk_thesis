import torch
import numpy as np
t = torch.arange(0, 10).reshape(2, 5)
print(t)

indices = torch.from_numpy(np.array([2, 4]))

t1 = torch.index_select(t, 1, indices)
print(t1)

t1 = torch.gather(t, 0, indices)
print(t1)
