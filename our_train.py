"""train TransRAC model """
from platform import node
import os
from LSPloader import MyData
from TransRAC import TransferModel
from our_looping import train_loop

# CUDA environment
N_GPU = 4
device_ids = [i for i in range(N_GPU)]
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


# # # we pick out the fixed frames from raw video file, and we store them as .npz file
# # # we currently support 64 or 128 frames
# data root path
root_path = r'/public/home/huhzh/LSP_dataset/LLSP_npz(64)/'

train_video_dir = 'train'
train_label_dir = 'train.csv'
valid_video_dir = 'valid'
valid_label_dir = 'valid.csv'

checkpoint = './checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth'
config = './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'

# TransRAC model checkpoint
lastckpt = 'checkpoint/ours/228_0.5419.pt'

NUM_FRAME = 64
# multi scales(list). we currently support 1,4,8 scale.
SCALES = [1, 4, 8]

train_dataset = MyData(root_path, train_video_dir, train_label_dir, num_frame=NUM_FRAME)
valid_dataset = MyData(root_path, valid_video_dir, valid_label_dir, num_frame=NUM_FRAME)
my_model = TransferModel(config=config, checkpoint=checkpoint, num_frames=NUM_FRAME, scales=SCALES, OPEN=False)
NUM_EPOCHS = 200
LR = 1e-5
BATCH_SIZE = 32

train_loop(NUM_EPOCHS, my_model, train_dataset, valid_dataset, train=True, valid=True,
           batch_size=BATCH_SIZE, lr=LR, saveckpt=True, ckpt_name='ours', log_dir='ours', device_ids=device_ids,
           lastckpt=lastckpt, mae_error=False)
