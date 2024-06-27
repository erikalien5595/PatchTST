import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from mamba_ssm import Mamba
from utils.tools import RevIN


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs=configs
        if self.configs.revin==1:
            if configs.is_cluster:
                self.revin_layer = RevIN(self.configs.enc_in_cluster)  # 聚类后的cluster内的channel数
            else:
                self.revin_layer = RevIN(self.configs.enc_in)  # 原始的channel数

        self.lin1=torch.nn.Linear(self.configs.seq_len, self.configs.d_model)
        self.dropout1=torch.nn.Dropout(self.configs.dropout)
        self.mamba = Mamba(d_model=self.configs.d_model,d_state=self.configs.d_state,d_conv=self.configs.dconv,expand=self.configs.e_fact)
        self.linear_head=torch.nn.Linear(self.configs.d_model,self.configs.pred_len)


    def forward(self, x):
        if self.configs.revin == 1:
            x = self.revin_layer(x, 'norm')
        else:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x /= stdev

        x = torch.permute(x, (0, 2, 1))
        x = self.lin1(x)
        x_res1 = x
        x = self.dropout1(x)
        x = self.mamba(x)
        x = self.linear_head(x+x_res1)
        x = x.permute(0, 2, 1)

        if self.configs.revin == 1:
            x = self.revin_layer(x, 'denorm')
        else:
            x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))
            x = x + (means[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))

        return x


if __name__=='__main__':
    batch, length, dim = 2, 64, 16
    x = torch.randn(batch, length, dim).to("cuda")
    model = Mamba(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=dim,  # Model dimension d_model
        d_state=16,  # SSM state expansion factor
        d_conv=4,  # Local convolution width
        expand=2,  # Block expansion factor
    ).to("cuda")
    y = model(x)
    print(123)
    assert y.shape == x.shape
