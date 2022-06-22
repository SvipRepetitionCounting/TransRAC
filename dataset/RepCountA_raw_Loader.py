"""
If you don't want to pre-process the RepCount dataset, 
you can use this script to load data from raw video(.mp4).
The trainning speed is slower and the memory cost more.
"""

import os
import os.path as osp
import numpy as np
import math
import cv2
from torch.utils.data import Dataset
import torch
import csv
import kornia
from .label_norm import normalize_label
import torchvision.transforms as transforms

class MyData(Dataset):

    def __init__(self, root_path, video_path, label_path, num_frame):
        self.root_path = root_path
        self.video_path = video_path
        self.label_dir = os.path.join(root_path, label_path)
        self.video_dir = os.listdir(os.path.join(self.root_path, self.video_path))
        self.label_dict = get_labels_dict(self.label_dir)  # get all labels
        self.num_frame = num_frame

    def __getitem__(self, inx):
        video_file_name= self.video_dir[inx]
        file_path = os.path.join(self.root_path, self.video_path, video_file_name)
        video_rd = VideoRead(file_path, num_frames=self.num_frame)
        video_tensor = video_rd.crop_frame()
        video_frame_length = video_rd.frame_length
        video_tensor = video_tensor.transpose(0, 1)  # [64, 3, 224, 224] -> [ 3, 64, 224, 224]
        if video_file_name in self.label_dict.keys():
            time_points = self.label_dict[video_file_name]
            label = preprocess(video_frame_length, time_points, num_frames=self.num_frame)
            label = torch.tensor(label)
            return [video_tensor, label]
        else:
            print(video_file_name, 'not exist')
            return
        
    def __len__(self):
        """返回数据集的大小"""
        return len(self.video_dir)


class VideoRead:
    def __init__(self, video_path, num_frames):
        self.video_path = video_path
        self.frame_length = 0
        self.num_frames = num_frames

    def get_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        assert cap.isOpened()
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        self.frame_length = len(frames)
        return frames

    def crop_frame(self):
        """to crop frames to tensor
        return: tensor [64, 3, 224, 224]
        """
        frames = self.get_frame()  # frames: the all frames of video
        frames_tensor = []
        if self.num_frames <= len(frames):
            for i in range(self.num_frames):
                #  select 64 frames from total original frames, proportionally
                frame = frames[i * len(frames) // self.num_frames]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))  # [3, 224, 224]
                frame = transforms.ToTensor()(frame)
                frames_tensor.append(frame)

        else:  # if raw frames number lower than 64, padding it. 
            for i in range(self.frame_length):
                frame = frames[i]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))  # [ 3, 224, 224]
                frame = transforms.ToTensor()(frame)
                frames_tensor.append(frame)
            for i in range(self.num_frames - self.frame_length):
                frame = frames[self.frame_length - 1]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))  # [ 3, 224, 224]
                frame = transforms.ToTensor()(frame)
                frames_tensor.append(frame)
        Frame_Tensor=torch.as_tensor(np.stack(frames_tensor))   

        return Frame_Tensor


def get_labels_dict(path):
    labels_dict = {}
    check_file_exist(path)
    with open(path, encoding='utf-8') as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            cycle = [int(row[key]) for key in row.keys() if 'L' in key and row[key] != '']
            labels_dict[row['name']] = cycle

    return labels_dict

def preprocess(video_frame_length, time_points, num_frames):
    """
    process label(.csv) to density map label
    Args:
        video_frame_length: video total frame number, i.e 1024frames
        time_points: label point example [1, 23, 23, 40,45,70,.....] or [0]
        num_frames: 64
    Returns: for example list [0.1,0.8,0.1, .....]
    """
    new_crop = []
    for i in range(len(time_points)):  # frame_length -> 64
        item = min(math.ceil((float((time_points[i])) / float(video_frame_length)) * num_frames), num_frames - 1)
        new_crop.append(item)
    new_crop = np.sort(new_crop)
    label = normalize_label(new_crop, num_frames)

    return label

def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))

# # example
# root_dir = r'/p300/data/dataset/'
# video_dir = 'train'
# label_dir = 'train.csv'
# train_dataset = MyData(root_dir, video_dir, label_dir,64)
# trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

