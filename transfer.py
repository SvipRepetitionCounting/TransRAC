"""example for transfer"""
from mmcv import Config, DictAction
from mmaction.models import build_model
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
import torch
import torch.nn as nn

from torch.cuda.amp import autocast, GradScaler
class FCNet(nn.Module):
    def __init__(self, in_channel, hidden_channel, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super().__init__()
        self.conv1x1 = nn.Conv3d(in_channel, hidden_channel, kernel_size=(1, 1, 1), stride=1, padding=0)
        self.layer = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.LayerNorm(n_hidden_1),
            nn.Dropout(p=0.5, inplace=False),
            nn.ReLU(True),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True),
            nn.Dropout(p=0.5,inplace=False),
            nn.Linear(n_hidden_2, out_dim)
        )

    def forward(self, x):
        with autocast():
            x = self.conv1x1(x)
            x = x.flatten(start_dim=1)
            x = self.layer(x)
            return x


class TransferModel(nn.Module):
    def __init__(self, config, checkpoint):
        super(TransferModel, self).__init__()
        self.config = config
        self.checkpoint = checkpoint
        self.backbone = self.load_model()
        self.transfer = FCNet(768, 32, 25088, 4096, 1024, 32)

    def load_model(self):
        cfg = Config.fromfile(self.config)
        model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model, self.checkpoint, map_location='cpu')
        backbone = model.backbone
        print('--------- backbone loaded ------------')
        return backbone

    def forward(self, x):
        with autocast():
            feature = self.backbone(x)
            x = self.transfer(feature)
            # print("output dim:", x.shape)
            return x


# config = './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'
# checkpoint = './checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth'

# root_dir = r'D:\人体重复运动计数\data\筛选数据集\ucf526'
# train_dir = 'video_agnostic'
# valid_dir = 'video_test'
# label_dir = r'D:\人体重复运动计数\data\筛选数据集\ucf526\annotations\repetition_label'
#
# NUM_FRAME = 64
# train_dataset = MyData(root_dir, train_dir, label_dir, num_frame=NUM_FRAME)
# trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# new_model = TransferModel(config, checkpoint)
# # [batch_size, channel, temporal_dim, height, width]
# # dummy_x = torch.rand(1, 3, 32, 224, 224)
# #
# # dummy_y=torch.rand(1,32)
# lossMSE = nn.MSELoss()
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, new_model.parameters()), lr=0.001)
# pbar = tqdm(trainloader, total=len(trainloader))
# for input, target in pbar:
#     new_model.train()
#     input = input.to(device)
#     density = target.to(device)
#     # print(target.shape)
#     # print(input.shape)
#     count = torch.sum(target, dim=1).round().to(device)
#     output = new_model(input)
#     predict_count = torch.sum(output, dim=1).round()
#     predict_density = output
#     loss = lossMSE(predict_density, density)
#     loss.backward()
#     optimizer.step()
#     print('loss', loss.item())
#     break
