"""train API of VST"""
import os

import torch

from LSPloader import MyData
from old_RepSwin import TransferModel
from our_looping_test import train_loop

N_GPU=1
device_ids = [i for i in range(N_GPU)]

root_dir = r'/group/DHZdata(64)/'
test_video_dir = 'test'
test_label_dir = 'test.csv'

config = './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'
checkpoint = './checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth'
lastckpt = '/p300/checkpoint/1112_2_99.pt'
NUM_FRAME = 64
SCALES=[1]
test_dataset = MyData(root_dir, test_video_dir, test_label_dir, num_frame=NUM_FRAME)
my_model = TransferModel(config=config, checkpoint=checkpoint, num_frames=NUM_FRAME,scales=SCALES)
NUM_EPOCHS = 1
LR = 1e-5
BATCH_SIZE = 1

train_loop(NUM_EPOCHS, my_model, test_dataset, inference=True,batch_size=BATCH_SIZE, paint=False,
           device_ids =device_ids, lastckpt=lastckpt)
