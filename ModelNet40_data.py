# Source: https://github.com/Strawberry-Eat-Mango/PCT_Pytorch

import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from download_ModelNet40 import download
import copy

def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def rotate(pointcloud):
    theta = np.pi * (np.random.uniform() - 0.5) / 3 # between -30deg and 30deg
    rot = np.zeros((3, 3))
    rot[:2, :2] = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rot[2, 2] = 1
    pointcloud = np.matmul(pointcloud, rot)
    return pointcloud

class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        if self.partition == 'train':
            pointcloud = self.data[item]
            np.random.shuffle(pointcloud)
            pointcloud = pointcloud[:self.num_points]
            pointcloud = translate_pointcloud(pointcloud)
            pointcloud = rotate(pointcloud)
        else:
            pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
