# -*- coding: UTF-8 -*-
# capg_data.py
# @Time     : 30/Mar/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
#

import os
import pickle
import re
import h5py

import torch
from torch.utils import data
import numpy as np
from scipy.io import loadmat


class CapgDataset(data.Dataset):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, gestures=8, sequence_len=1, train=True, transform=None):
        self.transform = transform
        self.train = train  # training set or test set

        root_path = os.path.join(os.sep, *os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-2])
        data_file_path = os.path.join(root_path, 'cache', 'capg.h5')
        if not os.path.isfile(data_file_path):
            print('data not exist, create a new h5 file')
            capg_data = _load_capg_all()
            train, test = _capg_train_test_split(capg_data, test_size=0.1)
            _save_capg_to_h5(train, test, data_file_path)
        else:
            print('data exist, load {} data from the file'.format('train' if train else 'test'))
            train, test = _load_capg_from_h5(data_file_path)

        if self.train:
            X, y = _prepare_data(train, required_gestures=gestures)
        else:
            X, y = _prepare_data(test, required_gestures=gestures)

        original_shape = X.shape
        if 1 == sequence_len:
            # reshape to (None*1000, 128)
            X = X.reshape(original_shape[0]*1000, 128)
            y = y.reshape((y.shape[0], 1)) * np.ones((1, 1000))
            y = y.flatten()
        else:
            # reshape to (None, seq_len, 128)
            seq_amount = int(1000/sequence_len)
            X = X.reshape(original_shape[0]*seq_amount, sequence_len, 128)
            y = y.reshape((y.shape[0], 1)) * np.ones((1, seq_amount))
            y = y.flatten()

        self.data, self.targets = X, y

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (X, target) where target is index of the target class.
        """
        x, y = self.data[index], int(self.targets[index])

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


def _prepare_data(data, required_gestures=8):
    """split train and test dataset
    """
    required_range = range(required_gestures)
    X = list()
    y = list()
    for i in required_range:
        amount = data[i].shape[0]
        X.append(data[i])
        y += [i for _ in range(amount)]
    X = np.concatenate(X, axis=0)
    X = X.astype(np.float32)
    y = np.asarray(y, dtype=np.int)

    return X, y


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


def _capg_train_test_split(data, test_size=0.1):
    train = dict()
    test = dict()

    for i in data.keys():
        shape = data[i].shape
        test_index = np.random.choice(shape[0], int(shape[0]*test_size), replace=False)
        train_index = np.delete(np.arange(shape[0]), test_index)
        test[i] = data[i][test_index]
        train[i] = data[i][train_index]

    return train, test


def _load_capg_from_h5(file_name):
    with h5py.File(file_name, 'r') as data_file:
        train = dict()
        train_grp = data_file['train']
        print()
        for i in train_grp.keys():
            train[int(i)] = train_grp[i].value

        test = dict()
        test_grp = data_file['test']
        for i in test_grp.keys():
            test[int(i)] = test_grp[i].value

    return train, test


def _save_capg_to_h5(train, test, file_name):
    with h5py.File(file_name, 'w') as data_file:
        train_grp = data_file.create_group('train')
        for gesture in train.keys():
            train_grp.create_dataset(str(gesture), data=train[gesture], dtype=float)

        test_grp = data_file.create_group('test')
        for gesture in test.keys():
            test_grp.create_dataset(str(gesture), data=test[gesture], dtype=float)

def _load_capg_data(db_name):
    # get the parent dir path
    now_path = os.path.join(os.sep, *os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-2])
    data_path = os.path.join(now_path, 'data', 'capg', db_name)
    mat_list = os.listdir(data_path)
    emg_data = _read_capg_mat_files(data_path, mat_list)
    
    return emg_data


def _read_capg_mat_files(path, mat_list):
    data = dict()

    for mat_name in mat_list:
        if not re.match(r'\S*[.]mat$', mat_name, flags=0):
            continue
        mat_path = os.path.join(path, mat_name)
        mat = loadmat(mat_path)
        gesture_index = mat['gesture'][0][0]
        if gesture_index in [100, 101]:
            continue
        
        if gesture_index not in data.keys():
            data[gesture_index] = list()
        data[gesture_index].append(mat['data'])

    return data


# def load_history(file_path):
#     with open(file_path, 'r') as history_file:
#         return pickle.load(history_file)


# def save_history(history, path):
#     with open(path, 'wb') as history_file:
#         pickle.dump(history, history_file)
