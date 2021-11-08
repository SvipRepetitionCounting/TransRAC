import csv
import os
import os.path as osp
import numpy as np
import math
import cv2
from torch.utils.data import Dataset
import kornia
import torch


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
        self.label_dict = get_labels_dict(self.label_path)  # get all labels
        self.num_frame = num_frame

    def __getitem__(self, inx):
        """获取数据集中的item  """
        video_file_name = self.video_dir[inx]
        file_path = os.path.join(self.video_path, video_file_name)
        video_rd = VideoRead(file_path, num_frames=self.num_frame)
        video_tensor = video_rd.crop_frame()
        #video_tensor = video_tensor.transpose(0, 1)  # [64, 3, 224, 224] -> [ 3, 64, 224, 224]
        if video_file_name in self.label_dict.keys():
            count = self.label_dict[video_file_name]
            label = generate_y(count, self.num_frame)
            label = torch.Tensor(label)
            return [video_tensor, label]
        else:
            label_dummy = torch.zeros([self.num_frame])
            print(video_file_name, 'not exist')
            return [video_tensor, label_dummy]

    def __len__(self):
        """返回数据集的大小"""
        return len(self.video_dir)


class VideoRead:
    def __init__(self, video_path, num_frames):
        self.video_path = video_path
        self.frame_length = 0
        self.num_frames = num_frames

    def get_frame(self):
        frames = []
        check_file_exist(self.video_path)
        cap = cv2.VideoCapture(self.video_path)
        assert cap.isOpened()
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
                frame = cv2.resize(frame, (112, 112))
                frames_tensor.append(frame)

        else:  # 当帧数不足时，补足帧数
            for i in range(self.frame_length):
                frame = frames[i]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (112, 112))
                frames_tensor.append(frame)
            for i in range(self.num_frames - self.frame_length):
                if len(frames) > 0:
                    frame = frames[-1]
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (112, 112))
                    frames_tensor.append(frame)
        frames_tensor = np.asarray_chkfinite(frames_tensor, dtype=np.uint8)
        try:
            frames_tensor = kornia.image_to_tensor(frames_tensor, keepdim=False).div(255.0)
        except:
            print('video error :', self.video_path)
            raise
        return frames_tensor


def get_labels_dict(path):
    labels_dict = {}
    check_file_exist(path)
    with open(path, encoding='utf-8') as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            labels_dict[row['name']] = row['count']

    return labels_dict


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def generate_y(count, num_frames=64):
    """生成repnet的label   [4,4,4,4]"""
    return [count for i in range(num_frames)]
