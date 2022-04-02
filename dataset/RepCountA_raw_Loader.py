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

from label_norm import normalize_label


class MyData(Dataset):

    def __init__(self, root_path, video_path, label_path, num_frame,sample_rate=4):
        """
        :param root_path: 数据集根目录
        :param video_path: 视频子目录
        """
        self.root_path = root_path
        self.video_path = video_path  # train or valid
        self.label_path = os.path.join(self.root_path, label_path)
        self.video_dir = os.listdir(os.path.join(self.root_path, self.video_path))
        self.label_dict = get_labels_dict(self.label_path) # get all labels
        self.num_frame = num_frame
        self.sample_rate=sample_rate

    def __getitem__(self, inx):
        video_file_name = self.video_dir[inx]
        file_path = os.path.join(self.root_path, self.video_path, video_file_name)
        video_rd = VideoSample(file_path, num_frames=self.num_frame)
        video_list = video_rd.crop_frame(sample_rate=self.sample_rate)

        video_frame_length = video_rd.frame_length

        # video_tensor = video_tensor.transpose(0, 1)  # [64, 3, 224, 224] -> [ 3, 64, 224, 224]
        if video_file_name  in self.label_dict.keys():
            print(video_file_name)
            time_crops = self.label_dict[video_file_name ]
            label_list = preprocess(video_frame_length, time_crops, num_frames=self.num_frame,sample_rate=self.sample_rate)
            if len(label_list)!=len(video_list):
                print("clip error:label clip {0},video clip {1}".format(len(label_list),len(video_list)))
            return video_list, label_list
        else:
            print(video_file_name , 'not exist')

        
    def __len__(self):
        """返回数据集的大小"""
        return len(self.video_dir)


class VideoSample:

    def __init__(self, video_path, num_frames):
        self.video_path = video_path
        self.frame_length = 0
        self.num_frames = num_frames

    def get_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        # assert cap.isOpened()
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))  # [64, 3, 224, 224]
            frames.append(frame)
        cap.release()
        self.frame_length = len(frames)
        return frames

    def crop_frame(self,sample_rate:int=4) -> list:
        """返回视频帧的切片tensor"""
        frames = self.get_frame()
        frames = frames[::sample_rate]  # 全视频采样
        self.frame_length=len(frames)
        if self.frame_length%self.num_frames!=0:  # 补成64的倍数帧
            zeropadding=np.zeros((self.num_frames-self.frame_length%self.num_frames,3,224,224))
            frames.append(zeropadding)
            self.frame_length=len(frames)
        
        frames_list=[]
        for i in range(0,self.frame_length-self.num_frames,self.num_frames):
            clip=np.array([frames[i:i+self.num_frames]]).reshape(self.num_frames,3,224,224)
            # clip=kornia.image_to_tensor(np.asarray_chkfinite(clip, dtype=np.uint8),keepdim=False).div(255.0)
            clip=torch.tensor(clip).div(255.0).transpose(0,1)
            # [64, 3, 224, 224] -> [ 3, 64, 224, 224]
            frames_list.append(clip)

        return frames_list


def get_labels_dict(path):
    labels_dict = {}
    check_file_exist(path)
    with open(path, encoding='utf-8') as f:
        f_csv = csv.DictReader(f)
        for row in f_csv:
            cycle = [int(row[key]) for key in row.keys() if 'L' in key and row[key] != '']
            labels_dict[row['name']] = cycle

    return labels_dict

def preprocess(length, crops, num_frames,sample_rate=4) ->list:
    new_crop = [math.ceil((float((crops[i]))/sample_rate)) for i in range(len(crops))]
    new_crop = np.sort(new_crop)
    print(length)
    label = normalize_label(new_crop, length)
    labels=[torch.tensor(label[i:i+num_frames]) for i in range(0,length-num_frames,num_frames)]

    return labels

def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))

# # example
# root_dir = r'/p300/data/dataset/'
# video_dir = 'train'
# label_dir = 'train.csv'
# train_dataset = MyData(root_dir, video_dir, label_dir,64)
# trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

