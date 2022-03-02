import csv
import math
import os
import os.path as osp

import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset


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
        video_tensor, frames_length = get_frames(file_path)  # [f,c,224,224]
        if video_file_name in self.label_dict.keys():
            count = self.label_dict[video_file_name]['count']
            cycle = self.label_dict[video_file_name]['cycle']
            cycle = preprocess(frames_length, cycle, num_frames=self.num_frame)
            y_length, y_onehot = generate_label(cycle, self.num_frame)
            y_length = torch.FloatTensor(y_length)
            y_onehot = torch.FloatTensor(y_onehot)

            return video_tensor, y_length, y_onehot
        else:
            label_dummy = torch.zeros([self.num_frame])
            print(video_file_name, 'not exist')
            return video_tensor, label_dummy

    def __len__(self):
        """返回数据集的大小"""
        return len(self.video_dir)


def get_frames(npz_path):
    with np.load(npz_path) as data:
        frames = data['imgs']
        frames = zoom(frames, (1, 1, 112 / 224, 112 / 224))
        frames_length = data['fps'].item()
        frames = torch.FloatTensor(frames)
        frames -= 127.5
        frames /= 127.5
    return frames, frames_length


def get_labels_dict(path):
    labels_dict = {}
    check_file_exist(path)
    with open(path, encoding='utf-8') as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            cycle = [int(row[key]) for key in row.keys() if 'L' in key and row[key] != '']
            if not row['count']:
                print(row['name'] + 'error')
            else:
                peroid_info = {'count': int(row['count']), 'cycle': cycle}
                labels_dict[row['name'].split('.')[0] + str('.npz')] = peroid_info

    return labels_dict


def generate_label(frames, num_frames=64):
    """
    frames:[2,5,5,8,9,11]
    num_frames: 12
    dong
    y1 = [0. 0. 3. 3. 3. 4. 4. 4. 4. 4. 4. 4. 4]
    y2 = [0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1]

    hu
    y1 =[0, 0, 3, 3, 3, 3, 3, 3, 0, 2, 2, 0]
    y2 = [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0]
    """
    y1 = [0 for i in range(num_frames)]  # 坐标轴长度，即帧数
    y2 = [0 for i in range(num_frames)]
    i = 0
    while i < len(frames):
        j = i + 1
        for k in range(frames[i], frames[j]):
            if frames[j] - frames[i] != 0:
                y1[k] = frames[j] - frames[i]
                y2[k] = 1
            else:
                y1[k] = 1
                y2[k] = 1
        i += 2
    y1 = np.array(y1).reshape(num_frames, -1)
    y2 = np.array(y2).reshape(num_frames, -1)
    return y1, y2


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
    if len(new_crop) % 2 != 0:
        print('label process error')
        raise

    return new_crop


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))
