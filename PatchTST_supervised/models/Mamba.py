import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from mamba_ssm import Mamba
from utils.tools import RevIN
from layers.Mamba_EncDec import Encoder, EncoderLayer


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs=configs
        if self.configs.revin==1:
            if configs.is_cluster:
                self.revin_layer = RevIN(self.configs.enc_in_cluster)  # 聚类后的cluster内的channel数
            else:
                self.revin_layer = RevIN(self.configs.enc_in)  # 原始的channel数
        self.ch_ind = configs.ch_ind
        print(f'self.ch_ind={self.ch_ind}')
        is_flip = 0 if self.ch_ind==1 else 1
        self.mamba1 = Mamba(d_model=configs.d_model, d_state=configs.d_state, d_conv=configs.dconv,
                            expand=configs.e_fact)
        self.mamba2 = Mamba(d_model=configs.d_model, d_state=configs.d_state, d_conv=configs.dconv,
                            expand=configs.e_fact)
        # if self.ch_ind == 1:
        #     self.mamba1 = Mamba(d_model=1, d_state=configs.d_state, d_conv=configs.dconv,
        #                     expand=configs.e_fact)
        #     self.mamba2 = Mamba(d_model=1, d_state=configs.d_state, d_conv=configs.dconv,
        #                     expand=configs.e_fact)
        # else:
        #     self.mamba1 = Mamba(d_model=configs.d_model, d_state=configs.d_state, d_conv=configs.dconv,
        #                         expand=configs.e_fact)
        #     self.mamba2 = Mamba(d_model=configs.d_model, d_state=configs.d_state, d_conv=configs.dconv,
        #                         expand=configs.e_fact)
        self.encoder = Encoder(
            [
                EncoderLayer(
                        self.mamba1, self.mamba2, configs.d_model, configs.d_ff, dropout=configs.dropout,
                        activation=configs.activation, is_flip=is_flip,
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)  # conv_layers=None
        )

        self.lin1 = torch.nn.Linear(self.configs.seq_len, self.configs.d_model)
        self.linear_head = torch.nn.Linear(self.configs.d_model, self.configs.pred_len)

    def forward(self, x):  # the original dimension of `x` is (B, L, D)
        # normalization
        if self.configs.revin == 1:
            x = self.revin_layer(x, 'norm')
        else:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x /= stdev
        B, L, D = x.shape
        x = torch.permute(x, (0, 2, 1))  # channel mixing: (B, D, L)
        if self.ch_ind == 1:
            x = torch.reshape(x, (B * D, 1, L))  # channel independent: (B * D, 1, L)
        x = self.lin1(x)  # Embedding: CD-(B, D, E), CI-(B * D, 1, E)
        x = self.encoder(x)
        x = self.linear_head(x)  # CD-(B, D, T), CI-(B * D, 1, T)
        if self.ch_ind == 1:
            x = torch.reshape(x, (-1, D, self.configs.pred_len))  # (B, D, T)
        x = x.permute(0, 2, 1)  # (B, T, D)

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
