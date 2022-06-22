import os

import numpy as np
import csv
import cv2
from torch.utils.data import Dataset, DataLoader
import kornia
import torch
import torchvision.transforms as transforms


class TestData(Dataset):
    def __init__(self, root_path, video_path, label_path, num_frame):
        self.root_path = root_path
        self.video_path = video_path
        self.label_dir = os.path.join(root_path, label_path)
        self.video_dir = os.listdir(os.path.join(self.root_path, self.video_path))
        self.label_dict = get_labels_dict(self.label_dir)  # get all labels
        self.num_frame = num_frame

    def __getitem__(self, inx):
        video_name= self.video_dir[inx]
        file_path = os.path.join(self.root_path, self.video_path, video_name)
        video_rd = VideoRead(file_path, num_frames=self.num_frame)
        video_tensor = video_rd.crop_frame()
        video_frame_length = video_rd.frame_length
        video_tensor = video_tensor.transpose(0, 1)  # [64, 3, 224, 224] -> [ 3, 64, 224, 224]
        count = self.label_dict[video_name]
        count = torch.tensor(count)

        return video_tensor, count
    def __len__(self):
        """返回数据集的大小"""
        return len(self.video_dir)


class VideoRead:
    def __init__(self, video_path, num_frames=64):
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
        """to crop frames to tensor [64, 3, 224, 224]"""
        frames = self.get_frame()  # frames: the all frames of video
        frames_tensor = []
        if self.num_frames <= len(frames):
            for i in range(self.num_frames):
                #  select 64 frames from total original frames, proportionally
                frame = frames[i * len(frames) // self.num_frames]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))  # [64, 3, 224, 224]
                frame = transforms.ToTensor()(frame)
                frames_tensor.append(frame)

        else:   # if raw frames number lower than 64, padding it. 
            for i in range(self.frame_length):
                frame = frames[i]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))  # [64, 3, 224, 224]
                frame = transforms.ToTensor()(frame)
                frames_tensor.append(frame)
            for i in range(self.num_frames - self.frame_length):
                frame = frames[self.frame_length - 1]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))  # [64, 3, 224, 224]
                frame = transforms.ToTensor()(frame)
                frames_tensor.append(frame)
        # frames_tensor = np.asarray_chkfinite(frames_tensor, dtype=np.uint8)
        # frames_tensor = kornia.image_to_tensor(frames_tensor, keepdim=False).div(255.0)
        Frame_Tensor=torch.as_tensor(np.stack(frames_tensor))   

        return Frame_Tensor


def get_labels_dict(path):
    labels_dict = {}
    check_file_exist(path)
    with open(path, encoding='utf-8') as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            cycle = [int(row[key]) for key in row.keys() if 'L' in key and row[key] != '']
            if not row['count']:
                print(row['name']+'error')
            else:
                labels_dict[row['name']] = int(row['count'])

    return labels_dict


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not os.path.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))




