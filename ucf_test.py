import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from ucf_loader import TestData
from tqdm import tqdm, trange
from old_RepSwin import TransferModel
from my_tools import paint_smi_matrixs,density_map

device_ids = [0]
device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


def train_loop(n_epochs, model,test_set, inference=True, batch_size=1, lastckpt=None, paint=False, device_ids=[0]):
    device = torch.device("cuda:" + str(device_ids[0]) if torch.cuda.is_available() else "cpu")
    currEpoch = 0
    testloader = DataLoader(test_set, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=20)
    model = nn.DataParallel(model.to(device), device_ids=device_ids)

    if lastckpt != None:
        print("loading checkpoint")
        checkpoint = torch.load(lastckpt)
        currEpoch = checkpoint['epoch']

        model.load_state_dict(checkpoint['state_dict'], strict=False)

        del checkpoint

    for epoch in tqdm(range(currEpoch, n_epochs + currEpoch)):
        testOBO = []
        testMAE = []
        predCount=[]
        Count=[]
        ACC=[]
        if inference:
            with torch.no_grad():
                batch_idx = 0
                pbar = tqdm(testloader, total=len(testloader))
                for input, target in pbar:
                    model.eval()
                    acc = 0
                    input = input.to(device)
                    count = target.to(device)
                    output, sim_matrix = model(input, epoch)
                    predict_count = torch.sum(output, dim=1).round()

                    mae = torch.sum(torch.div(torch.abs(predict_count - count), count + 1e-1)) / \
                            predict_count.flatten().shape[0]  # mae

                    gaps = torch.sub(predict_count, count).reshape(-1).cpu().detach().numpy().reshape(-1).tolist()
                    for item in gaps:
                        if abs(item) <= 1:
                            acc += 1
                    OBO = acc / predict_count.flatten().shape[0]
                    testOBO.append(OBO)
                    MAE = mae.item()
                    testMAE.append(MAE)

                    predCount.append(predict_count.item())
                    Count.append(count.item())
                    print('predict count :{0}, groundtruth :{1}'.format(predict_count.item(), count.item()))
                    #if predict_count.item() == count.item() and MAE < 0.2:
                    #    density_map(output, count.item(), batch_idx)
                    # if paint:
                    #     paint_smi_matrixs(sim_matrix,batch_idx)
                    batch_idx += 1

        print("MAE:{0},OBO:{1}".format(np.mean(testMAE),np.mean(testOBO)))


ckpt = r'/p300/checkpoint/1113_128_99.pt'

root_dir = '/group/ucf526_valid/'
video_dir = 'valid'
label_dir = 'ucfval.csv'
config = './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'
checkpoint = './checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth'
NUM_FRAME = 128
SCALES=[1]
test_set = TestData(root_dir, video_dir, label_dir, num_frame=NUM_FRAME)
my_model = TransferModel(config=config, checkpoint=checkpoint, num_frames=NUM_FRAME,scales=SCALES)
NUM_EPOCHS = 1
LR = 1e-5
BATCH_SIZE = 1
train_loop(NUM_EPOCHS, my_model, test_set, inference=True,batch_size=BATCH_SIZE, paint=False,
           device_ids =device_ids, lastckpt=ckpt)

