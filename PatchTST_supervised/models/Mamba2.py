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
            self.revin_layer = RevIN(self.configs.enc_in)
        self.ch_ind = configs.ch_ind
        self.is_permute = 0
        # Embedding
        self.enc_embedding = DataEmbeddingInverted(configs.seq_len, configs.d_model, configs.embed_type, configs.freq,
                                                    configs.dropout)
        if configs.is_cluster==1:
            self.encoder_dict = nn.ModuleDict()
            self.ch_ind = {}
            for cluster,value in configs.label_dict.items():
                ch_ind = 0 if configs.sra_dict[cluster] > self.configs.corr_threshold else 1
                self.ch_ind[cluster] = ch_ind
                print("111", cluster, value, f'ch_ind={ch_ind}')
                self.mamba1 = Mamba(d_model=configs.d_model, d_state=configs.d_state, d_conv=configs.dconv,
                                    expand=configs.e_fact)
                self.mamba2 = Mamba(d_model=configs.d_model, d_state=configs.d_state, d_conv=configs.dconv,
                                    expand=configs.e_fact)

                if ch_ind == 1 and self.is_permute == 1:
                    self.mamba1 = Mamba(d_model=1, d_state=configs.d_state, d_conv=configs.dconv,
                                        expand=configs.e_fact)
                    self.mamba2 = Mamba(d_model=1, d_state=configs.d_state, d_conv=configs.dconv,
                                        expand=configs.e_fact)

                self.encoder_dict[str(cluster)] = Encoder(
                    [
                        EncoderLayer(
                            self.mamba1, self.mamba2, configs.d_model, configs.d_ff, dropout=configs.dropout,
                            activation=configs.activation,
                        ) for _ in range(configs.e_layers)
                    ],
                    norm_layer=torch.nn.LayerNorm(configs.d_model)  # conv_layers=None
                )
        else:
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
                            activation=configs.activation,
                    ) for _ in range(configs.e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model)  # conv_layers=None
            )
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

        if self.configs.is_cluster == 1:
            # Embedding for CD: (B, L, D) -> (B, D, E), CI: (B, L, D)  -> (B * D, 1, E)
            # enc_in = self.enc_embedding(x_enc, None)  # covariates (e.g. timestamp) can be also embedded as tokens
            enc_in = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g. timestamp) can be also embedded as tokens
            enc_in_tf_channel = enc_in[:, D:, :]

            enc_out = torch.zeros((B, D, self.configs.d_model)).float().to(self.configs.device)
            for cluster, encoder in self.encoder_dict.items():
                cluster = int(cluster)
                enc_in_cluster = enc_in[:, self.configs.label_dict[cluster], :]
                _, D_in_cluster, _ = enc_in_cluster.shape
                if x_mark_enc is not None:
                    enc_in_cluster = torch.cat([enc_in_cluster, enc_in_tf_channel], dim=1)
                    _, D_in_cluster_total, _ = enc_in_cluster.shape
                # enc_out : (B, D_in_cluster, E)
                if self.ch_ind[cluster] == 0:
                    enc_out[:, self.configs.label_dict[cluster], :] = encoder(enc_in_cluster)[:, :D_in_cluster, :]
                else:
                    enc_in_cluster = enc_in_cluster.reshape(
                        (B * D_in_cluster_total, 1, self.configs.d_model))  # channel independent: (B * D, 1, E)
                    if self.is_permute == 0:
                        enc_out[:, self.configs.label_dict[cluster], :] = encoder(enc_in_cluster).reshape((-1, D_in_cluster_total, self.configs.d_model))[:, :D_in_cluster, :]
                    else:  # is_permute == 1
                        enc_in_cluster = enc_in_cluster.permute(0, 2, 1)  # channel independent: (B * D, E, 1) or (B * (D+D1), E, 1)
                        enc_out[:, self.configs.label_dict[cluster], :] = encoder(
                            enc_in_cluster).permute(0, 2, 1).reshape((-1, D_in_cluster_total, self.configs.d_model))[:, :D_in_cluster, :]
            dec_out = self.linear_head(enc_out).permute(0, 2, 1)  # (B, T, D)
        else:
            enc_in = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g. timestamp) can be also embedded as tokens
            if self.ch_ind == 1:
                if x_mark_enc is None:
                    enc_in = torch.reshape(enc_in, (B * D, 1, self.configs.d_model))  # (B * D, 1, E)
                else:
                    enc_in = torch.reshape(enc_in, (B * (D+D1), 1, self.configs.d_model))  # (B * D, 1, E)
                if self.is_permute == 1:
                    enc_in = enc_in.permute(0, 2, 1)  # channel independent: (B * D, E, 1) or (B * (D+D1), E, 1)
            # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
            enc_out = self.encoder(enc_in)
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
