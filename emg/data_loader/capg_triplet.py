# -*- coding: utf-8 -*-
# capg_triplet.py
# @Time     : 28/Jun/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
#


import os
import re
import h5py
import math

import torch
from torch.utils.data import DataLoader, Dataset
from scipy.io import loadmat
import numpy as np
import random


class CapgTriplet(Dataset):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, gesture_list, sequence_len,
                 frame_x=False, train=True, transform=None):

        self.transform = transform
        self.train = train  # training set or test set
        self.seq_length = sequence_len
        self.frame_x = frame_x  # x is frame format or not

        root_path = os.path.join(os.sep, *os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-2])
        processed_data = os.path.join(root_path, 'data', 'capg-processed')
        train_data_path = os.path.join(processed_data, 'train.h5')
        test_data_path = os.path.join(processed_data, 'test.h5')
        if os.path.isfile(train_data_path) and os.path.isfile(test_data_path):
            print('processed capg data exist, load {} data from the file'.format('train' if train else 'test'))
            train_set, test_set = _load_capg_from_h5(train_data_path, test_data_path)
        else:
            print('processed capg data not exist, create new h5 files')
            os.mkdir(processed_data)
            capg_data = _load_capg_all()
            train_set, test_set = _capg_train_test_split(capg_data, test_size=0.1)
            _save_capg_to_h5(train_set, test_set, train_data_path, test_data_path)

        if self.train:
            self._emg_data = _prepare_data(train_set, gesture_list)
        else:
            self._emg_data = _prepare_data(test_set, gesture_list)

        self.scale = 380000
        self.gesture_amount = len(gesture_list)

    def __getitem__(self, index):
        """根据index返回triplet的data pairs
        Args:
            index (int): Index

        Returns:
            tuple: (X, target) where target is index of the target class.
        """
        true_index = index // self.scale
        gesture_idx = true_index % self.gesture_amount
        amount = len(self._emg_data[gesture_idx])
        anchor_idx, positive_idx = np.random.choice(amount, 2, replace=False)
        rest_gestures = list(self._emg_data.keys())
        rest_gestures.remove(gesture_idx)
        negative_gesture_idx = np.random.choice(rest_gestures)
        amount = len(self._emg_data[negative_gesture_idx])
        negative_idx = np.random.choice(amount)

        anchor = self._emg_data[gesture_idx][anchor_idx]
        positive = self._emg_data[gesture_idx][positive_idx]
        negative = self._emg_data[negative_gesture_idx][negative_idx]

        triplet_emg = [anchor, positive, negative]

        for i in range(len(triplet_emg)):
            emg_segment = triplet_emg[i]
            start = np.random.randint(0, 1000 - self.seq_length)
            end = start + self.seq_length
            emg_segment = emg_segment[start:end]  # now x.shape is (N, 128)

            if self.frame_x:
                # x shape is (N, 128), reshape to frame format: (N, 16, 8)
                if self.seq_length == 1:
                    # used for 2d cnn
                    emg_segment = emg_segment.reshape((1, 16, 8))
                else:
                    # used for 3d cnn
                    emg_segment = emg_segment.reshape((1, emg_segment.shape[0], 16, 8))
            elif self.seq_length == 1:
                emg_segment = emg_segment[0]
            triplet_emg[i] = torch.tensor(emg_segment, dtype=torch.float)
        target = torch.tensor([gesture_idx])
        return triplet_emg, target

    def __len__(self):
        return self.scale * self.gesture_amount


def _prepare_data(raw_data: dict, gesture_list):
    """
    """
    emg_data = {gesture_idx: raw_data[gesture_idx] for gesture_idx in gesture_list}
    return emg_data


def _load_capg_all():
    # load three databases of capg
    dba = _load_capg_data('dba')
    dbb = _load_capg_data('dbb')
    dbc = _load_capg_data('dbc')

    capg = {gesture_index: None for gesture_index in range(20)}
    for i in dba.keys():
        capg_index = i - 1
        capg[capg_index] = np.asarray(dba[i] + dbb[i])

    for i in dbc.keys():
        capg_index = i + 7
        capg[capg_index] = np.asarray(dbc[i])

    return capg


def _capg_train_test_split(raw_data, test_size=0.1):
    train_set = dict()
    test_set = dict()

    for i in raw_data.keys():
        shape = raw_data[i].shape
        test_index = np.random.choice(shape[0], int(shape[0] * test_size), replace=False)
        train_index = np.delete(np.arange(shape[0]), test_index)
        test_set[i] = raw_data[i][test_index]
        train_set[i] = raw_data[i][train_index]

    return train_set, test_set


def _load_capg_from_h5(train_file_path, test_file_path):
    with h5py.File(train_file_path, 'r') as train_file:
        train_set = dict()
        train_grp = train_file['train']
        for i in train_grp.keys():
            train_set[int(i)] = train_grp[i][()]

    with h5py.File(test_file_path, 'r') as test_file:
        test_set = dict()
        test_grp = test_file['test']
        for i in test_grp.keys():
            test_set[int(i)] = test_grp[i][()]

    return train_set, test_set


def _save_capg_to_h5(train_set, test_set, train_file_path, test_file_path):
    with h5py.File(train_file_path, 'w') as train_file:
        train_grp = train_file.create_group('train')
        for gesture in train_set.keys():
            train_grp.create_dataset(str(gesture), data=train_set[gesture], dtype=float)

    with h5py.File(test_file_path, 'w') as test_file:
        test_grp = test_file.create_group('test')
        for gesture in test_set.keys():
            test_grp.create_dataset(str(gesture), data=test_set[gesture], dtype=float)


def _load_capg_data(db_name):
    # get the parent dir path
    now_path = os.path.join(os.sep, *os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-2])
    data_path = os.path.join(now_path, 'data', 'capg', db_name)
    mat_list = os.listdir(data_path)
    emg_data = _read_capg_mat_files(data_path, mat_list)

    return emg_data


def _read_capg_mat_files(path, mat_list):
    mat_data = dict()

    for mat_name in mat_list:
        if not re.match(r'\S*[.]mat$', mat_name, flags=0):
            continue
        mat_path = os.path.join(path, mat_name)
        mat = loadmat(mat_path)
        gesture_index = mat['gesture'][0][0]
        if gesture_index in [100, 101]:
            continue

        if gesture_index not in mat_data.keys():
            mat_data[gesture_index] = list()
        mat_data[gesture_index].append(mat['data'])

    return mat_data


if __name__ == '__main__':
    # test pytorch data loader
    test_list = list(range(8))
    train_data = CapgTriplet(test_list,
                             sequence_len=10,
                             frame_x=True,
                             train=True)
    print(len(train_data))
    dataloader = DataLoader(train_data, batch_size=2,
                            shuffle=True, num_workers=4)
    for i_batch, (xi, yi) in enumerate(dataloader):
        # print(i_batch)
        print(len(xi))
        for i in xi:
            print(i.size())
        print(yi)
        break
