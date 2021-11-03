import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import logging
import cv2
from torch.utils.data import Dataset, DataLoader
import kornia
import torch
from label_norm import normalize_label


class MyData(Dataset):

    def __init__(self, root_path, video_path, label_path, num_frame):
        """
        :param root_path: 数据集根目录
        :param video_path: 视频子目录
        """
        self.root_path = root_path
        self.video_path = os.path.join(self.root_path, video_path)  # train or valid
        self.label_path = os.path.join(self.root_path, label_path)
        self.video_dir = os.listdir(self.video_path)
        self.label_dict = LabelRd(self.label_path).get_labels_dict()  # get all labels
        self.num_frame = num_frame

    def __getitem__(self, inx):
        """获取数据集中的item  """
        # try:
        video_file_name = self.video_dir[inx]
        file_path = os.path.join(self.video_path, video_file_name)
        # print(video_file_name)
        video_rd = VideoRead(file_path, num_frames=self.num_frame)
        video_tensor = video_rd.crop_frame()
        video_frame_length = video_rd.frame_length
        video_tensor = video_tensor.transpose(0, 1)  # [64, 3, 224, 224] -> [ 3, 64, 224, 224]
        if video_file_name in self.label_dict.keys():
            time_crops = self.label_dict[video_file_name]
            label = preprocess(video_frame_length, time_crops, num_frames=self.num_frame)
            label = torch.Tensor(label)
            return [video_tensor, label]
        else:
            label_dummy = torch.zeros([self.num_frame])
            print(video_file_name, 'not exist')
            return [video_tensor, label_dummy]

        # except Exception as e:
        #     logging.error("get item error : %s " % e)

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
        """to get video frame tensor"""
        frames = self.get_frame()
        frames_tensor = []
        if self.num_frames <= self.frame_length:
            for i in range(1, self.num_frames + 1):
                frame = frames[i * len(frames) // self.num_frames - 1]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))  # [64, 3, 224, 224]
                # frame = transform(frame).unsqueeze(0)
                frames_tensor.append(frame)

        else:  # 当帧数不足时，补足帧数
            for i in range(self.frame_length):
                frame = frames[i]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224,224))  # [64, 3, 224, 224]
                # frame = transform(frame).unsqueeze(0)
                frames_tensor.append(frame)
            for i in range(self.num_frames - self.frame_length):
                if len(frames)>0:
                    frame = frames[-1]
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (224, 224))  # [64, 3, 224, 224]
                    frames_tensor.append(frame)
        frames_tensor = np.asarray_chkfinite(frames_tensor, dtype=np.uint8)
        frames_tensor = kornia.image_to_tensor(frames_tensor, keepdim=False).div(255.0)
        return frames_tensor


class LabelRd:

    def __init__(self, path):
        self.path = path

    def get_labels_dict(self):
        labels_dict = {}
        with open(self.path, encoding='utf-8') as f:
            f_csv = csv.DictReader(f)
            for row in f_csv:
                cycle = [int(row[key]) for key in row.keys() if 'L' in key and row[key] != '']
                labels_dict[row['name']] = cycle

        return labels_dict


def preprocess(length, crops, num_frames):
    """
    original cycle list to label
    Args:
        length: frame_length
        crops: label point example [6 ,31, 44, 54] or [0]
        num_frames: 64
    Returns: [6,31,31,44,44,54]
    """
    new_crop = []
    for i in range(len(crops)):  # frame_length -> 64
        item = min(math.ceil((float((crops[i])) / float(length)) * num_frames), num_frames - 1)
        new_crop.append(item)
    new_crop = np.sort(new_crop)
    label = normalize_label(new_crop, num_frames)

    return label


# #
# root_dir = r'D:\人体重复运动计数\LSPdataset'
# train_video_dir = 'train'
# train_label_dir = 'train.csv'
# valid_video_dir = 'valid'
# valid_label_dir = 'valid.csv'
# train_dataset = MyData(root_dir, train_video_dir, train_label_dir, 32)
# valid_dataset = MyData(root_dir, valid_video_dir, valid_label_dir, num_frame=32)
# trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True,num_workers=4)
# validloader = DataLoader(valid_dataset, batch_size=4, shuffle=True,num_workers=4)
#
# print(len(trainloader))
#
#
# for batch_idx, batch_data in enumerate(trainloader):
#     # print(batch_data[0].shape)
#     # print(batch_data[1].shape)
#     print(batch_idx)
#     if not batch_data:
#         print('error')
# print('over')
