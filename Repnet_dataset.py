import csv
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
        video_tensor = get_frames(file_path)  # [f,c,224,224]
        if video_file_name in self.label_dict.keys():
            count = self.label_dict[video_file_name]['count']
            cycle = self.label_dict[video_file_name]['cycle']
            y_length, y_onehot = generate_label(cycle, self.num_frame)
            y_length = torch.FloatTensor(y_length)
            y_onehot = torch.FloatTensor(y_onehot)

            return [video_tensor, y_length, y_onehot]
        else:
            label_dummy = torch.zeros([self.num_frame])
            print(video_file_name, 'not exist')
            return [video_tensor, label_dummy]

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
    return frames


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


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def generate_label(frames, num_frames=64):
    """
    frames:[2,5,5,8,9,12]
    num_frames: 64
    y1 = [0. 0. 3. 3. 3. 4. 4. 4. 4. 4. 4. 4. 4. 0. 0. 0.]
    y2 = [0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0.]
    """
    y1 = np.zeros(num_frames, dtype=float)  # 坐标轴长度，即帧数
    y2 = np.zeros(num_frames, dtype=float)
    # for i in range(0,y_length.size,2):
    for i in range(0, len(frames), 2):
        x_a = frames[i]
        x_b = frames[i + 1]
        if i + 2 < num_frames.size:
            if x_b == frames[i + 2]:
                if x_a != x_b:
                    x_b -= 1
                elif frames[i + 2] != frames[i + 3]:
                    frames[i + 2] += 1
                else:
                    continue
        p = x_b - x_a + 1
        for j in range(x_a, x_b + 1):
            y1[j] = p
            y2[j] = 1
    return y1, y2
