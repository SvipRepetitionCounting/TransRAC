"""test TransRAC model"""
import os
## if your data is .mp4 form, please use RepCountA_raw_Loader.py
# from dataset.RepCountA_raw_Loader import MyData
## if your data is .npz form, please use RepCountA_Loader.py. It can speed up the training
from dataset.RepCountA_Loader import MyData
from models.TransRAC import TransferModel
from testing.test_looping import test_loop

N_GPU = 1
device_ids = [i for i in range(N_GPU)]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# # # we pick out the fixed frames from raw video file, and we store them as .npz file
# # # we currently support 64 or 128 frames
# data root path
root_path = r'/public/home/huhzh/LSP_dataset/LLSP_npz(64)/'

test_video_dir = 'test'
test_label_dir = 'test.csv'

# video swin transformer pretrained model and config
config = './configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'
checkpoint = './pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth'

# TransRAC trained model checkpoint, we will upload soon.
lastckpt = None

NUM_FRAME = 64
# multi scales(list). we currently support 1,4,8 scale.
SCALES = [1, 4, 8]
test_dataset = MyData(root_path, test_video_dir, test_label_dir, num_frame=NUM_FRAME)
my_model = TransferModel(config=config, checkpoint=checkpoint, num_frames=NUM_FRAME, scales=SCALES, OPEN=False)
NUM_EPOCHS = 1

test_loop(NUM_EPOCHS, my_model, test_dataset, lastckpt=lastckpt)
