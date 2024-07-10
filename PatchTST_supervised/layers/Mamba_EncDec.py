import concurrent.futures
import threading

import torch.nn as nn
import torch.nn.functional as F
import torch
from mamba_ssm import Mamba
from layers.SelfAttention_Family import FullAttention, AttentionLayer
class EncoderLayer(nn.Module):
    def __init__(self, mamba1, mamba2, dropout=0.05, is_flip=1):
        super(EncoderLayer, self).__init__()
        self.mamba1 = mamba1
        self.mamba2 = mamba2
        self.dropout = nn.Dropout(dropout)
        self.is_flip = is_flip
    def forward(self, x):
        x_res = x
        x = self.dropout(x)
        if self.is_flip:
            new_x = self.mamba1(x) + self.mamba2(x.flip(dims=[1])).flip(dims=[1])  # B, D, E
        else:
            new_x = self.mamba1(x)
        x = new_x + x_res

        return x


class Encoder(nn.Module):
    def __init__(self, mamba_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.norm = norm_layer

    def forward(self, x):
        # x: [B, D, E]
        for mamba_layer in self.mamba_layers:
            x = mamba_layer(x)  # (B, D, E)

        # if self.norm is not None:
        #     x = self.norm(x)

        return x