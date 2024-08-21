import torch
import torch.nn as nn
from mamba_ssm import Mamba
from utils.tools import RevIN
from layers.Mamba_EncDec import Encoder, EncoderLayer
from layers.Embed import DataEmbeddingInverted


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
        self.is_permute = 0
        # Embedding
        self.enc_embedding = DataEmbeddingInverted(configs.seq_len, configs.d_model, configs.embed_type, configs.freq,
                                                    configs.dropout)
        self.mamba1 = Mamba(d_model=configs.d_model, d_state=configs.d_state, d_conv=configs.dconv,
                            expand=configs.e_fact)
        self.mamba2 = Mamba(d_model=configs.d_model, d_state=configs.d_state, d_conv=configs.dconv,
                            expand=configs.e_fact)
        if self.ch_ind==1 and self.is_permute ==1:
            self.mamba1 = Mamba(d_model=1, d_state=configs.d_state, d_conv=configs.dconv,
                                expand=configs.e_fact)
            self.mamba2 = Mamba(d_model=1, d_state=configs.d_state, d_conv=configs.dconv,
                                expand=configs.e_fact)
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

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):  # the original dimension of `x` is (B, L, D)
        # normalization
        if self.configs.revin == 1:
            x_enc = self.revin_layer(x_enc, 'norm')
        else:
            means = x_enc.mean(1, keepdim=True)#.detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)#.detach()
            x_enc /= stdev

        B, L, D = x_enc.shape
        _, _, D1 = x_mark_enc.shape
        # x_mark_enc = None
        # print(f'D={x_enc[:3]},D1={x_mark_enc[:3]}')
        # B: batch_size;    E: d_model;
        # L: seq_len;       T: pred_len;
        # D: number of variate (tokens), can also include covariates

        # Embedding for CD: (B, L, D) -> (B, D, E), CI: (B, L, D)  -> (B * D, 1, E)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g. timestamp) can be also embedded as tokens
        if self.ch_ind == 1:
            if x_mark_enc is None:
                enc_out = torch.reshape(enc_out, (B * D, 1, self.configs.d_model))  # channel independent: (B * D, 1, E)
            else:
                enc_out = torch.reshape(enc_out, (B * (D+D1), 1, self.configs.d_model))  # channel independent: (B * (D+D1), 1, E)
            if self.is_permute ==1:
                enc_out = enc_out.permute(0, 2, 1)  # channel independent: (B * D, E, 1) or (B * (D+D1), E, 1)

        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out = self.encoder(enc_out)
        # B D E -> B D T -> B T D
        if self.ch_ind == 1 and self.is_permute == 1:
            enc_out = enc_out.permute(0, 2, 1)  # 变回 (B*D, 1, E) or (B * (D+D1), 1, E)
        dec_out = self.linear_head(enc_out)  # CD:(B, D, T), CI:(B * D, 1, T)
        if self.ch_ind == 1:
            if x_mark_enc is None:
                dec_out = torch.reshape(dec_out, (-1, D, self.configs.pred_len))  # (B, D, T)
            else:
                dec_out = torch.reshape(dec_out, (-1, D+D1, self.configs.pred_len))  # (B, D+D1, T)
        dec_out = dec_out.permute(0, 2, 1)[:, :, :D]  # (B, T, D) filter the covariates

        if self.configs.revin == 1:
            dec_out = self.revin_layer(dec_out, 'denorm')
        else:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.configs.pred_len:, :]  # [B, L, D]


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
