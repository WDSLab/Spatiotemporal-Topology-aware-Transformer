import torch
import torch.nn as nn
import torch.nn.functional as F
from SToT.modules.Transformer_EncDec import Encoder, EncoderLayer, series_decomp, s_series_decomp, TS_EncoderLayer
from SToT.modules.SelfAttention_Family import AttentionLayer, FlowAttention, AutoCorrelationLayer, AutoCorrelation
from SToT.modules.Embed import STDataEmbedding_inverted_geo_embed
import numpy as np


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.node_num = configs.node_num
        self.d_model = configs.d_model
        self.device = configs.device

        # Embedding
        self.enc_embedding = STDataEmbedding_inverted_geo_embed(configs.seq_len, configs.d_model, configs.root_path, configs.seq_len, configs.embed, configs.freq,
                                                    configs.dropout)

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = s_series_decomp(kernel_size)

        # Encoder-only architecture
        self.encoder = Encoder(
            [

                TS_EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    AttentionLayer(     # flow-attention
                        FlowAttention(attention_dropout=configs.dropout), configs.node_num+6, configs.n_heads),
                    configs.d_model,
                    configs.pred_len,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        self.trend_projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        self.affine_weight = nn.Parameter(torch.ones(1, 1, configs.enc_in))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, configs.enc_in))

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            # encoder input init & normalization
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
            x_enc = x_enc * self.affine_weight + self.affine_bias

        B, S, N = x_enc.shape   # B S T

        # B: batch_size;    D: d_model; 
        # S: seq_len;       N: node dim;

        # Embedding
        # B S N -> B N D
        enc_out = self.enc_embedding(x_enc, x_mark_enc) 

        # B N D -> B N D                
        seasonal_part, trend_part, attns = self.encoder(enc_out, attn_mask=None, origin_trend=torch.zeros(B, enc_out.size(1), self.d_model).to(self.device))

        # B N E -> B N S -> B S N 
        seasonal_part = self.projector(seasonal_part).permute(0, 2, 1).contiguous()[:, :, :N] # filter the covariates
        trend_part = self.trend_projector(trend_part).permute(0, 2, 1).contiguous()[:, :, :N] # filter the covariates
        dec_out = seasonal_part + trend_part
        # dec_out = seasonal_part + trend_part + trend

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out - self.affine_bias
            dec_out = dec_out / (self.affine_weight + 1e-10)
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]