import torch
import torch.nn as nn
from video_swin_transformer import SwinTransformer3D
import numpy as np
from tensorboardX import SummaryWriter

# model = SwinTransformer3D()
# # print(model)
#
# dummy_x = torch.rand(2, 3, 128, 224, 224)
# logits = model(dummy_x)
#
# print('-------logits:( [batch_size, channel, temporal_dim, height, width] )----------',logits.shape)  # [2, 768, 32, 7, 7]


'''
load the pretrained weight
1. git clone https://github.com/SwinTransformer/Video-Swin-Transformer.git
2. move all files into ./Video-Swin-Transformer

'''
from mmcv import Config, DictAction
from mmaction.models import build_model
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

# config = './configs/recognition/swin/swin_base_patch244_window1677_sthv2.py'
# checkpoint = './checkpoints/swin_base_patch244_window1677_sthv2.pth'
config = './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'
checkpoint = './checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth'

cfg = Config.fromfile(config)
model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, checkpoint, map_location='cpu')


class Net(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super().__init__()
        self.conv3d = nn.Conv3d(768, 64, kernel_size=(1, 1, 1), stride=1, padding=0)
        self.layer = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU(True),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True),
            nn.Linear(n_hidden_2, out_dim)
        )

    def forward(self, x):
        x = self.conv3d(x)
        x = x.flatten(start_dim=1)
        x = self.layer(x)
        return x


# [batch_size, channel, temporal_dim, height, width]
dummy_x = torch.rand(1, 3, 64, 224, 224)
dummy_y=torch.rand(1,32)
lossMSE = nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
backbone = model.backbone
# feat = backbone(dummy_x)
# batch_size, hidden_dim, temporal_dim, height, width = feat.shape
# print('-------feature stretch:( [batch_size, channel, temporal_dim, height, width] )----------', feat.shape)  # [1, 768, 16, 7, 7]
cls_head=Net(100352,4096,512,32)
feat=backbone(dummy_x)
output=cls_head(feat)


print(output.shape)
loss=lossMSE(output,dummy_y)
print(loss)
loss.backward()
optimizer.step()
# print(model)
# model.cls_head=Net()
# # SwinTransformer3D without cls_head
# backbone = model.backbone
# # print(backbone)
# # [batch_size, hidden_dim, temporal_dim/2, height/32, width/32]
# feat = backbone(dummy_x)
# print('-------feature stretch:( [batch_size, channel, temporal_dim, height, width] )----------', feat.shape)
# batch_size, hidden_dim, temporal_dim, height, width = feat.shape
# feat1 = feat.mean(dim=[2, 3, 4])  # [batch_size, hidden_dim]
# input_dim = hidden_dim
# print("input features dim:[{0},{1}] ".format(batch_size, input_dim))
# net = Net(input_dim, 512, 128, 16)
# output = net(feat1)
# print("output dim:", output.shape)
