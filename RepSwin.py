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
from timm.models.layers import trunc_normal_


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
            scores = scores.masked_fill_(attn_mask, -np.inf)
        scores = self.softmax(scores)
        scores = self.dropout(scores)  # [B,head, F, F]
        # context = torch.matmul(scores, v)  # output
        return scores  # [B,head,F, F]


class Similarity_matrix(nn.Module):

    def __init__(self, num_heads=4, model_dim=512):
        super().__init__()

        # self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.input_size = 512
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
    ''' 全连接预测网络 '''

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
        self.apply(self._init_weights)

    def forward(self, x):
        # x = x.flatten(start_dim=1)
        x = self.layers(x)
        return x

    def _init_weights(self, m):
        """ 权重初始化 """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class TransferModel(nn.Module):
    def __init__(self, config, checkpoint, num_frames):
        super(TransferModel, self).__init__()
        self.num_frames = num_frames
        self.config = config
        self.checkpoint = checkpoint
        self.scales = [1, 4, 8]  # 多尺度
        self.backbone = self.load_model()
        self.Replication_padding1 = nn.ReplicationPad3d((0, 0, 0, 0, 1, 1))
        self.Replication_padding2 = nn.ReplicationPad3d((0, 0, 0, 0, 2, 2))
        self.Replication_padding4 = nn.ReplicationPad3d((0, 0, 0, 0, 4, 4))
        self.conv3D = nn.Conv3d(in_channels=768,
                                out_channels=512,
                                kernel_size=3,
                                padding=(3, 1, 1),
                                dilation=(3, 1, 1))
        self.bn1 = nn.BatchNorm3d(512)
        self.SpatialPooling = nn.MaxPool3d(kernel_size=(1, 7, 7))

        self.sims = Similarity_matrix()
        self.conv3x3 = nn.Conv2d(in_channels=4 * len(self.scales),  # num_head*scale_num
                                 out_channels=32,
                                 kernel_size=3,
                                 padding=1)

        self.bn2 = nn.BatchNorm2d(32)

        self.dropout1 = nn.Dropout(0.25)
        self.input_projection = nn.Linear(self.num_frames * 32, 512)  #
        self.ln1 = nn.LayerNorm(512)

        self.transEncoder = TransEncoder(d_model=512, n_head=4, dropout=0.2, dim_ff=512, num_layers=1)
        self.FC = Prediction(512, 512, 256, 1)  # 1104 输出为1
        self.apply(self._init_weights)

    def load_model(self):
        cfg = Config.fromfile(self.config)
        model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model, self.checkpoint, map_location='cpu')
        backbone = model.backbone
        print('--------- backbone loaded ------------')
        return backbone

    def forward(self, x, epoch):
        with autocast():
            batch_size, c, num_frames, h, w = x.shape
            multi_scales = []
            for scale in self.scales:
                if scale == 4:
                    x = self.Replication_padding2(x)
                    crops = [x[:, :, i:i + scale, :, :] for i in
                             range(0, self.num_frames - scale + scale // 2 * 2, max(scale // 2, 1))]
                elif scale == 8:
                    x = self.Replication_padding4(x)
                    crops = [x[:, :, i:i + scale, :, :] for i in
                             range(0, self.num_frames - scale + scale // 2 * 2, max(scale // 2, 1))]
                else:
                    crops = [x[:, :, i:i + 1, :, :] for i in range(0, self.num_frames)]

                slice = []
                if epoch < 50:
                    with torch.no_grad():
                        for crop in crops:
                            crop = self.backbone(crop)  # ->[batch_size, 768, scale/2(up), 7, 7]  帧过vst
                            slice.append(crop)
                else:
                    for crop in crops:
                        crop = self.backbone(crop)  # ->[batch_size, 768, scale/2(up), 7, 7]  帧过vst
                        slice.append(crop)

                x_scale = torch.cat(slice, dim=2)  # ->[b,768,f,size,size]
                x_scale = F.relu(self.bn1(self.conv3D(x_scale)))  # ->[b,512,f,7,7]
                # print(x_scale.shape)
                x_scale = self.SpatialPooling(x_scale)  # ->[b,512,f,1,1]
                x_scale = x_scale.squeeze(3).squeeze(3)  # -> [b,512,f]
                x_scale = x_scale.transpose(1, 2)  # -> [b,32,512]
                x_scale = x_scale.reshape(batch_size, self.num_frames, -1)  # -> [b,f,512]

                # -------- similarity matrix ---------
                x_sims = F.relu(self.sims(x_scale, x_scale, x_scale))  # -> [b,4,f,f]
                multi_scales.append(x_sims)

            x = torch.cat(multi_scales, dim=1)  # [B,4*scale_num,f,f]
            x_matrix = x
            x = F.relu(self.bn2(self.conv3x3(x)))  # [b,32,f,f]
            x = self.dropout1(x)

            x = x.permute(0, 2, 3, 1)  # [b,f,f,32]
            # --------- transformer encoder ------
            x = x.flatten(start_dim=2)  # ->[b,f,32*f]
            x = F.relu(self.input_projection(x))  # ->[b,f, 512]
            x = self.ln1(x)

            x = x.transpose(0, 1)  # [f,b,512]
            x = self.transEncoder(x)  #
            x = x.transpose(0, 1)  # ->[b,f, 512]

            # x = x.flatten(1)  # ->[b,f*512]
            x = self.FC(x)  # ->[b,f,1]

            x = x.view(batch_size, self.num_frames)

            return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

# root_dir = r'D:\人体重复运动计数\LSPdataset'
# train_video_dir = 'train'
# train_label_dir = 'train.csv'
# valid_video_dir = 'valid'
# valid_label_dir = 'valid.csv'
#
# config = './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'
# checkpoint = './checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth'
#
# dummy_x = torch.rand(2, 3, 8, 224, 224)
# NUM_FRAME = 8
# # train_dataset = MyData(root_dir, train_video_dir, train_label_dir, num_frame=NUM_FRAME)
# my_model = TransferModel(config=config, checkpoint=checkpoint,num_frames=NUM_FRAME)
# # trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
# # for input, target in trainloader:
# #     print('input',input.shape)
# #     out=my_model(input)
# #     break
# out=my_model(dummy_x)
# print(out.shape)
# count = torch.sum(out, dim=1).round()
# print(count)
