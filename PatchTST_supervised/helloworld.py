import torch
import numpy as np
mask = torch.zeros((4, 4))
mask[1,2] = 1
mask[2,3] = 1
print(mask)
mask = torch.where(mask == 0, torch.tensor([-float('inf')]), torch.tensor([0.0]))
print(mask)

import pycatch22
tsData = np.asarray([[1,2,4,3], [1,2,4,3]]) # (or more interesting data!)
print(tsData)
# print(pycatch22.CO_f1ecac(tsData))
print(pycatch22.catch22_all(tsData,catch24=True,short_names=True))