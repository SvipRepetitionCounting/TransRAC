"""Repnet based on swin-t"""
from mmcv import Config
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
import torch
import torch.nn as nn
import math
from torch.cuda.amp import autocast
import numpy as np
from LSPloader import MyData
from torch.utils.data import DataLoader
import torch.nn.functional as F


class attention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, scale=64, att_dropout=None):
        super().__init__()
        # self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(att_dropout)
        self.scale = scale

    def forward(self, q, k, v, attn_mask=None):
        # q: [B, head, F, model_dim]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.scale)  # [B,Head, F, F]
        if attn_mask:
            # 给需要mask的地方设置一个负无穷
            scores = scores.masked_fill_(attn_mask, -np.inf)
        scores = self.softmax(scores)
        scores = self.dropout(scores)  # [B,head, F, F]
        # context = torch.matmul(scores, v)  # output
        return scores  # [B,head,F, F]


class Similarity_matrix(nn.Module):

    def __init__(self, num_heads=8, model_dim=512):
        super().__init__()

        # self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.input_size = 128
        self.linear_q = nn.Linear(self.input_size, model_dim)
        self.linear_k = nn.Linear(self.input_size, model_dim)
        self.linear_v = nn.Linear(self.input_size, model_dim)

        self.attention = attention(att_dropout=0)
        # self.out = nn.Linear(model_dim, model_dim)
        # self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        # 残差连接
        batch_size = query.size(0)
        # dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        # linear projection
        query = self.linear_q(query)  # [B,F,model_dim]
        key = self.linear_k(key)
        value = self.linear_v(value)
        # split by heads
        # [B,F,model_dim] ->  [B,F,num_heads,per_head]->[B,num_heads,F,per_head]
        query = query.view(batch_size, -1, num_heads, self.model_dim // self.num_heads).transpose(1, 2)
        key = key.view(batch_size, -1, num_heads, self.model_dim // self.num_heads).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, self.model_dim // self.num_heads).transpose(1, 2)
        # similar_matrix :[B,H,F,F ]
        matrix = self.attention(query, key, value, attn_mask)

        return matrix


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x


class TransEncoder(nn.Module):
    def __init__(self, d_model, n_head, dim_ff, dropout=0.0, num_layers=1):
        super(TransEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, 0.1, 64)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=n_head,
                                                   dim_feedforward=dim_ff,
                                                   dropout=dropout,
                                                   activation='relu')
        encoder_norm = nn.LayerNorm(d_model)
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

    def forward(self, src):
        src = self.pos_encoder(src)
        e_op = self.trans_encoder(src)
        return e_op


class Prediction(nn.Module):
    def __init__(self, input_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Prediction, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, n_hidden_1),
            nn.LayerNorm(n_hidden_1),
            nn.Dropout(p=0.25, inplace=False),
            nn.ReLU(True),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True),
            nn.Dropout(p=0.25, inplace=False),
            nn.Linear(n_hidden_2, out_dim)
        )

    def forward(self, x):
        # x = x.flatten(start_dim=1)
        x = self.layers(x)
        return x


class TransferModel(nn.Module):
    def __init__(self, config, checkpoint):
        super(TransferModel, self).__init__()
        self.config = config
        self.checkpoint = checkpoint
        self.backbone = self.load_model()
        self.sims = Similarity_matrix()
        self.ln1 = nn.Linear(768 * 49, 128)
        self.dropout1 = nn.Dropout(0.25)
        self.ln2 = nn.Linear(8 * 1 * 32, 512)  # head* len(scale) * f
        self.transEncoder = TransEncoder(d_model=512, n_head=4, dropout=0.2, dim_ff=512, num_layers=1)
        self.FC = Prediction(32 * 512, 1024, 256, 32)

    def load_model(self):
        cfg = Config.fromfile(self.config)
        model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model, self.checkpoint, map_location='cpu')
        backbone = model.backbone
        print('--------- backbone loaded ------------')
        return backbone

    def forward(self, x):
        with autocast():
            b, c, t, h, w = x.shape
            scales = [1]
            multi_scales = []
            for scale in scales:
                if scale != 1:
                    padding_size = scale // 2
                    zero_padding = torch.zeros(b, c, padding_size, h, w)  # temporal zero padding
                    x = torch.cat((zero_padding, x, zero_padding), dim=2)
                    crops = [x[:, :, i:i + scale, :, :] for i in
                             range(0, 32 - scale + padding_size * 2, max(scale // 2, 1))]
                else:
                    crops = [x[:, :, i:i + scale, :, :] for i in range(0, 32)]
                # print('padding shape: ', x.shape)
                slice = []
                for crop in crops:
                    crop = self.backbone(crop)  # ->[batch_size, 768, scale/2, height/32, width/32]  帧过vst
                    crop = crop.transpose(1, 2)  # ->[B, scale/2, 768,size,size]
                    crop = crop.flatten(start_dim=2)  # ->[B, scale/2, 768*7*7]
                    crop = F.relu(self.ln1(crop))  # [B, scale/2,128]  降维 768->128
                    slice.append(crop)
                x_scale = torch.cat(slice, dim=1)
                # print(x_scale.shape)
                # -------- similarity matrix ---------
                x_sims = self.sims(x_scale, x_scale, x_scale)  # -> [b,multi-head,F,F]
                multi_scales.append(x_sims)

            x = torch.cat(multi_scales, dim=1)
            # --------- transformer encoder ------
            x = x.transpose(1, 2)  # -> [B,F,head* scale,F]
            x = x.flatten(start_dim=2)  # ->[b,f,head*scale*f]
            x = F.relu(self.ln2(x))  # ->[b,f, 512]
            x=self.dropout1(x)
            x = self.transEncoder(x)  # ->[b,f, 512]
            x = x.flatten(1)  # ->[b,f*512]
            x = self.FC(x)  # ->[b,32]

            return x

