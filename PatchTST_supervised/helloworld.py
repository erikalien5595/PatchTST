from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from layers.PatchTST_layers import series_decomp
import torch

x = np.arange(0, 100, 1)

data = 1 + 3 * np.sin(2* np.pi / 7 * x) + 0.25 * x + np.random.randn(100)
data2 = 5 - 4 * np.cos(2* np.pi / 12 * x) - 0.1 * x + np.random.randn(100)*0.8
data = np.column_stack([data, data2])
L, D = data.shape
print(data.shape)
plt.figure()
plt.plot(data)
plt.show()
plt.figure()
data = torch.Tensor(data).reshape((1, -1, D))
decomp_module = series_decomp(31)
season, trend = decomp_module(data)
season = season.numpy().reshape(-1, D)
trend = trend.numpy().reshape(-1, D)
print(season.shape, trend.shape)
plt.plot(season)
plt.show()
plt.figure()
plt.plot(trend)
plt.show()
# 得到趋势、周期性、随机变量的数据输出
# print(rd.trend)
# print(rd.seasonal)
# print(rd.resid)
exit()

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