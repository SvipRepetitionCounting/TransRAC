"""train API of VST"""
import os

import torch

from LSPloader import MyData
from RepSwin import TransferModel
from our_looping import train_loop

N_GPU=1
device_ids = [i for i in range(N_GPU)]

root_dir = r'/p300/data/LSPdataset'
# root_dir = r'D:\人体重复运动计数\LSPdataset'
train_video_dir = 'train'
train_label_dir = 'train.csv'
valid_video_dir = 'valid'
valid_label_dir = 'valid.csv'

config = './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'
checkpoint = './checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth'
lastckpt = '/p300/checkpoint/1105_1_64_9.pt'
NUM_FRAME = 64

train_dataset = MyData(root_dir, train_video_dir, train_label_dir, num_frame=NUM_FRAME)
valid_dataset = MyData(root_dir, valid_video_dir, valid_label_dir, num_frame=NUM_FRAME)
my_model = TransferModel(config=config, checkpoint=checkpoint, num_frames=NUM_FRAME)
NUM_EPOCHS = 100
LR = 1e-5
BATCH_SIZE = 1

train_loop(NUM_EPOCHS, my_model, train_dataset, valid_dataset, train=True, valid=True,
           batch_size=BATCH_SIZE, lr=LR, saveCkpt=True, ckpt_name='1108_1',log_dir='scalar1108_1', lastCkptPath=lastckpt,
           device_ids=device_ids)
