import torch
import numpy as np
a = torch.randn([32,336, 1],device="cuda")
b = np.zeros([32, 336, 1])
b = a.detach().cpu().numpy()
print(b)