import concurrent.futures
import threading

import torch.nn as nn
import torch.nn.functional as F
import torch
from mamba_ssm import Mamba
from layers.SelfAttention_Family import FullAttention, AttentionLayer
class EncoderLayer(nn.Module):
    def __init__(self, mamba1, mamba2, d_model, d_ff, dropout=0.05, activation='gelu', is_flip=1):
        super(EncoderLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mamba1 = mamba1
        self.mamba2 = mamba2
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.is_flip = is_flip

    def forward(self, x):
        x_res = x
        x = self.dropout(x)
        # if self.is_flip:
        #     new_x = self.mamba1(x) + self.mamba2(x.flip(dims=[1])).flip(dims=[1])  # B, D, E
        # else:
        #     new_x = self.mamba1(x)
        new_x = self.mamba1(x) + self.mamba2(x.flip(dims=[1])).flip(dims=[1])  # B, D, E
        x = new_x + x_res
        # x = self.dropout(new_x) + x_res

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, mamba_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.norm = norm_layer

    def forward(self, x):
        # x: [B, D, E]
        for mamba_layer in self.mamba_layers:
            x = mamba_layer(x)  # (B, D, E)

        if self.norm is not None:
            x = self.norm(x)

        return x