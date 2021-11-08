"""our method VST"""
import os

import numpy as np
import torch
import torch.nn as nn
from mmcv.parallel import MMDataParallel
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm


torch.manual_seed(1)

def train_loop(n_epochs, model, train_set, valid_set, train=True, valid=True, batch_size=1, lr=1e-5,
               ckpt_name='ckpt', lastCkptPath=None, saveCkpt=False, log_dir='scalar', device_ids=[0]):
    device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
    currEpoch = 0
    trainloader = DataLoader(train_set, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=20)
    validloader = DataLoader(valid_set, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=20)
    model = MMDataParallel(model.to(device), device_ids=device_ids)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    lr_list = []
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.8)  # three step decay

    writer = SummaryWriter(log_dir=os.path.join('/p300/log/', log_dir))
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

    for epoch in tqdm(range(currEpoch, n_epochs + currEpoch)):
        trainLosses = []
        validLosses = []
        trainLoss1 = []
        validLoss1 = []
        trainOBO = []
        validOBO = []
        trainMAE = []
        validMAE = []

        if train:
            pbar = tqdm(trainloader, total=len(trainloader))
            batch_idx = 0
            for input, target in pbar:
                with autocast():
                    model.train()
                    optimizer.zero_grad()
                    acc = 0
                    input = input.to(device)
                    density = target.to(device)
                    count = torch.sum(target, dim=1).round().to(device)
                    output = model(input, epoch)
                    predict_count = torch.sum(output, dim=1).round()
                    predict_density = output
                    loss1 = lossMSE(predict_density, density)
                    loss2 = lossSL1(predict_count, count)
                    loss3 = torch.sum(torch.div(torch.abs(predict_count - count), count + 1e-1)) / \
                            predict_count.flatten().shape[0]  # mae
                    loss = loss1  # 1104: l1+l2

                    gaps = torch.sub(predict_count, count).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
                    for item in gaps:
                        if abs(item) <= 1:
                            acc += 1
                    OBO = acc / predict_count.flatten().shape[0]
                    trainOBO.append(OBO)
                    MAE = loss3.item()
                    trainMAE.append(MAE)

                    batch_loss = loss.item()
                    batch_loss1 = loss1.item()
                    trainLosses.append(batch_loss)
                    trainLoss1.append(batch_loss1)
                    batch_idx += 1
                    pbar.set_postfix({'Epoch': epoch,
                                      'loss_train': batch_loss,
                                      'Train MAE': MAE,
                                      'Train OBO ': OBO})

                    if batch_idx % 10 == 0:
                        writer.add_scalars('train/loss',
                                           {"loss": np.mean(trainLosses)},
                                           epoch * len(trainloader) + batch_idx)
                        # writer.add_scalars('train/loss1',
                        #                    {"loss1": np.mean(trainLoss1)},
                        #                    epoch * len(trainloader) + batch_idx)
                        writer.add_scalars('train/MAE',
                                           {"MAE": np.mean(trainMAE)},
                                           epoch * len(trainloader) + batch_idx)
                        writer.add_scalars('train/OBO',
                                           {"OBO": np.mean(trainOBO)},
                                           epoch * len(trainloader) + batch_idx)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        if valid:
            with torch.no_grad():
                batch_idx = 0
                pbar = tqdm(validloader, total=len(validloader))
                for input, target in pbar:
                    model.eval()
                    acc = 0
                    input = input.to(device)
                    density = target.to(device)
                    count = torch.sum(target, dim=1).round().to(device)

                    output = model(input, epoch)
                    predict_count = torch.sum(output, dim=1).round()
                    predict_density = output

                    loss1 = lossMSE(predict_density, density)
                    loss2 = lossSL1(predict_count, count)
                    loss3 = torch.sum(torch.div(torch.abs(predict_count - count), count + 1e-1)) / \
                            predict_count.flatten().shape[0]  # mae
                    loss = loss1  # 1104 loss=loss1+loss2

                    gaps = torch.sub(predict_count, count).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
                    for item in gaps:
                        if abs(item) <= 1:
                            acc += 1
                    OBO = acc / predict_count.flatten().shape[0]
                    validOBO.append(OBO)
                    MAE = loss3.item()
                    validMAE.append(MAE)
                    batch_loss = loss.item()
                    batch_loss1 = loss1.item()
                    validLosses.append(batch_loss)
                    validLoss1.append(batch_loss1)
                    batch_idx += 1
                    pbar.set_postfix({'Epoch': epoch,
                                      'loss_valid': batch_loss,
                                      'Valid MAE': MAE,
                                      'Valid OBO ': OBO})

                    writer.add_scalars('valid/loss', {"loss": np.mean(validLosses)},
                                       epoch * len(validloader) + batch_idx)
                    # writer.add_scalars('valid/loss1', {"loss1": np.mean(validLoss1)},
                    #                    epoch * len(validloader) + batch_idx)
                    writer.add_scalars('valid/OBO', {"OBO": np.mean(validOBO)},
                                       epoch * len(validloader) + batch_idx)
                    writer.add_scalars('valid/MAE',
                                       {"MAE": np.mean(validMAE)},
                                       epoch * len(trainloader) + batch_idx)

        scheduler.step()
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

        if saveCkpt and (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainLosses': trainLosses,
                'valLosses': validLosses,
                'model': model
            }
            torch.save(checkpoint,
                       '/p300/checkpoint/' + ckpt_name + '_' + str(epoch) + '.pt')

        writer.add_scalars('learning rate',
                           {"learning rate": optimizer.state_dict()['param_groups'][0]['lr']},
                           epoch)
