""""train of swin-transformer"""
import torch
import torch.nn as nn
from my_swin_transformer import SwinTransformer3D
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from LSPloader import MyData
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from transfer import TransferModel
from torch.cuda.amp import autocast, GradScaler
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
device_ids = [4,5,6, 7]
device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


def train_loop(n_epochs, model, train_set, valid_set, train=True, valid=True, batch_size=4, lr=1e-3,
               ckpt_name='ckpt',
               lastCkptPath=None, saveCkpt=False, log_dir='scalar'):
    """
    Args:
        n_epochs:
        model:
        train_set: dataset object
        valid_set: dataset object
        train:
        valid:
        batch_size:
        lr:
        ckpt_name: ckpt file name
        lastCkptPath: checkpoint file.pt
        saveCkpt: default False
        log_dir: tensorboard save path
        train Hu-swin-transformer
    Returns:

    """
    currEpoch = 0
    trainLosses = []
    validLosses = []
    trainOBO = []
    validOBO = []
    trainMAE = []
    validMAE = []
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    validloader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=4)
    model = MMDataParallel(model.to(device), device_ids=device_ids)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scaler = GradScaler()
    if lastCkptPath != None:
        print("loading checkpoint")
        checkpoint = torch.load(lastCkptPath)
        currEpoch = checkpoint['epoch']
        trainLosses = checkpoint['trainLosses']
        validLosses = checkpoint['valLosses']

        model.load_state_dict(checkpoint['state_dict'], strict=False)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        del checkpoint

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    lossMSE = nn.MSELoss()
    lossSL1 = nn.SmoothL1Loss()
    writer = SummaryWriter(log_dir=log_dir)
    for epoch in tqdm(range(currEpoch, n_epochs + currEpoch)):
        if train:
            pbar = tqdm(trainloader, total=len(trainloader))
            batch_idx = 0
            mean_loss = 0
            for input, target in pbar:
                with autocast():
                    model.train()
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    input = input.to(device)
                    density = target.to(device)
                    count = torch.sum(target, dim=1).round().to(device)
                    acc = 0
                    output = model(input)
                    predict_count = torch.sum(output, dim=1).round()
                    predict_density = output
                    loss1 = lossMSE(predict_density, density)
                    loss2 = lossSL1(predict_count, count)

                    loss = 5 * loss1 + loss2  # 1024: 5*l1+l2
                    # loss = loss1
                    gaps = torch.abs(torch.sub(predict_count, count)).reshape(-1).cpu().detach().numpy().reshape(
                        -1).tolist()
                    for item in gaps:
                        if item <= 1:
                            acc += 1
                    OBO = acc / batch_size
                    trainOBO.append(OBO)

                    MAE = torch.sum(torch.div(torch.abs(torch.sub(predict_count, count)), count)).item() / batch_size
                    trainMAE.append(MAE)

                    batch_loss = loss.item()
                    mean_loss += batch_loss
                    trainLosses.append(batch_loss)
                    batch_idx += 1
                    pbar.set_postfix({'Epoch': epoch,
                                      'loss_train': mean_loss / batch_idx,
                                      'Train MAE': MAE,
                                      'Train OBO ': OBO,
                                      'learning rate':optimizer.state_dict()['param_groups'][0]['lr']})
                    if batch_idx % 10 == 0:
                        writer.add_scalars('train/loss',
                                           {"mean_loss": np.mean(trainLosses)},
                                           epoch * len(trainloader) + batch_idx)
                        writer.add_scalars('train/OBO',
                                           {"OBO": np.mean(trainOBO)},
                                           epoch * len(trainloader) + batch_idx)
                        # writer.add_scalars('train/MAE',
                        #                    {"MAE": np.mean(trainMAE)},
                        #                    epoch * len(trainloader) + batch_idx)
                        writer.add_scalars('train/learning rate',
                                           {"learning rate": optimizer.state_dict()['param_groups'][0]['lr']},
                                           epoch * len(validloader) + batch_idx)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        if valid:
            with torch.no_grad():
                batch_idx = 0
                mean_loss = 0
                pbar = tqdm(validloader, total=len(validloader))
                for input, target in pbar:
                    model.eval()
                    torch.cuda.empty_cache()
                    acc = 0.0
                    input = input.to(device)
                    density = target.to(device)
                    count = torch.sum(target, dim=1).round().to(device)

                    output = model(input)
                    predict_count = torch.sum(output, dim=1).round()
                    predict_density = output

                    loss1 = lossMSE(predict_density, density)
                    loss2 = lossSL1(predict_count, count)

                    loss = 5 * loss1 + loss2
                    # loss = loss1
                    gaps = torch.sub(predict_count, count).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
                    for item in gaps:
                        if abs(item) <= 1:
                            acc += 1
                    OBO = acc / batch_size
                    validOBO.append(OBO)

                    MAE = torch.sum(torch.div(torch.abs(torch.sub(predict_count, count)), count)).item() / batch_size
                    validMAE.append(MAE)

                    batch_loss = loss.item()
                    mean_loss += batch_loss
                    validLosses.append(batch_loss)
                    batch_idx += 1
                    pbar.set_postfix({'Epoch': epoch,
                                      'loss_valid': (mean_loss / batch_idx),
                                      'Valid MAE': MAE,
                                      'Valid OBO ': OBO})

                    writer.add_scalars('valid/loss', {"mean_loss": np.mean(validLosses)},
                                       epoch * len(validloader) + batch_idx)
                    writer.add_scalars('valid/OBO', {"OBO": np.mean(validOBO)},
                                       epoch * len(validloader) + batch_idx)

        if saveCkpt and epoch % 50 == 0:
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainLosses': trainLosses,
                'valLosses': validLosses
            }
            torch.save(checkpoint,
                       'checkpoint/' + ckpt_name + '/' + ckpt_name + '_' + str(epoch) + '.pt')

    return trainLosses, validLosses


# root_dir = r'/root/video-swin-transformer-pytorch/Video-Swin-Transformer/tools/data'
# train_dir = 'video_agnostic'
# valid_dir = 'video_test'
# label_dir = r'/root/video-swin-transformer-pytorch/Video-Swin-Transformer/tools/data/repetition_label'

# root_dir = r'D:\人体重复运动计数\data\筛选数据集\ucf526'
# train_dir = 'video_agnostic'
# valid_dir = 'video_test'
# label_dir = r'D:\人体重复运动计数\data\筛选数据集\ucf526\annotations\repetition_label'

# root_dir = r'/p300/data/LSPdataset'
root_dir = r'D:\人体重复运动计数\LSPdataset'
train_video_dir = 'train'
train_label_dir = 'train.csv'
valid_video_dir = 'valid'
valid_label_dir = 'valid.csv'

config = './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'
checkpoint = './checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth'

NUM_FRAME = 32
train_dataset = MyData(root_dir, train_video_dir, train_label_dir, num_frame=NUM_FRAME)
valid_dataset = MyData(root_dir, valid_video_dir, valid_label_dir, num_frame=NUM_FRAME)
my_model = TransferModel(config=config, checkpoint=checkpoint)
NUM_EPOCHS = 50
LR = 1e-4

train_loss, valid_loss = train_loop(NUM_EPOCHS, my_model, train_dataset, valid_dataset, train=False, valid=True,
                                    batch_size=1, lr=LR, saveCkpt=True, ckpt_name='1028_lsp', log_dir='scalar1028_lsp')
