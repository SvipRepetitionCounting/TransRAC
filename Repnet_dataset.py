import csv
import os
import os.path as osp
import numpy as np
from torch.utils.data import Dataset
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
        video_tensor = get_frames(file_path)  # [f,c,224,224]
        if video_file_name in self.label_dict.keys():
            count = self.label_dict[video_file_name]
            label = generate_y(count, self.num_frame)
           
            label = torch.FloatTensor(label)
            return [video_tensor, label]
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
        frames_length=data['fps'].item()
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
                print(row['name']+'error')
            else:
                labels_dict[row['name'].split('.')[0]+str('.npz')] = int(row['count'])

    return labels_dict


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def generate_y(count, num_frames=64):
    """生成repnet的label   [4,4,4,4]"""
    y=[count for i in range(num_frames)]
    y=np.array(y).reshape(num_frames,-1)
    return y
