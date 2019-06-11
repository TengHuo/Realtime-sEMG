# -*- coding: UTF-8 -*-
# csl_data.py
# @Time     : 11/Jun/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
#

import os
import re
import math
import pandas as pd

from torch.utils.data import DataLoader, Dataset
from scipy.io import loadmat
import numpy as np


# def default_csl_loaders(model_args: dict):
#     seq_length = model_args['seq_length'] if 'seq_length' in model_args else 1
#     train_loader = DataLoader(CSLDataset(gesture=model_args['gesture_num'],
#                                           sequence_len=seq_length,
#                                           sequence_result=model_args['seq_result'],
#                                           frame_x=model_args['frame_input'],
#                                           test_mode=False,
#                                           train=True),
#                               batch_size=model_args['train_batch_size'],
#                               num_workers=4,
#                               shuffle=True)
#
#     val_loader = DataLoader(CSLDataset(gesture=model_args['gesture_num'],
#                                         sequence_len=seq_length,
#                                         sequence_result=model_args['seq_result'],
#                                         frame_x=model_args['frame_input'],
#                                         test_mode=False,
#                                         train=False),
#                             batch_size=model_args['val_batch_size'],
#                             num_workers=4,
#                             shuffle=False)
#
#     return train_loader, val_loader


class CSLDataset(Dataset):
    """An abstract class representing a Dataset.
    """

    def __init__(self, train=True, test_mode=False, transform=None):

        self.transform = transform
        self.train = train
        self.scale = 100

        root_path = os.path.join(os.sep, *os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-2])
        processed_data = os.path.join(root_path, 'data', 'csl-processed')
        train_data_path = os.path.join(processed_data, 'train.csv')
        test_data_path = os.path.join(processed_data, 'test.csv')
        if os.path.isfile(train_data_path) and os.path.isfile(test_data_path):
            print('processed csl data exist, load {} data from the file'.format('train' if train else 'test'))
            train_set, test_set = _load_csl_from_csv(train_data_path, test_data_path)
        else:
            print('processed csl data not exist, create new h5 files')
            # os.mkdir(processed_data)
            csl_data = _load_csl_all()
            train_set, test_set = _csl_train_test_split(csl_data, test_size=0.1)
            _save_csl_to_csv(train_set, test_set, train_data_path, test_data_path)

        gestures = ['gest0.mat', 'gest1.mat', 'gest2.mat', 'gest3.mat', 'gest4.mat']
        if self.train:
            X, y = _prepare_data(train_set, gesture_list=gestures)
        else:
            X, y = _prepare_data(test_set, gesture_list=gestures)

        if test_mode and self.train:
            # for test code, only use small part of data
            self.data, self.targets = X[:64], y[:64]
        else:
            self.data, self.targets = X, y

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (X, target) where target is index of the target class.
        """
        true_index = math.floor(index / self.scale)
        x_path = self.data[true_index]
        x_index = int(x_path[-1])
        x_path = x_path[0:-1]

        mat = loadmat(x_path)
        mat = mat['gestures'][x_index][0]
        x = mat.T

        # x.shape is (1000, 128)
        # start = np.random.randint(0, 1000 - self.seq_length)
        # end = start + self.seq_length
        # x = x[start:end]  # now x.shape is (N, 128)

        # if self.frame_x:
        #     # x shape is (N, 128), reshape to frame format: (N, 16, 8)
        #     if self.seq_length == 1:
        #         # used for 2d cnn
        #         x = x.reshape((1, 16, 8))
        #     else:
        #         # used for 3d cnn
        #         x = x.reshape((1, x.shape[0], 16, 8))
        # elif self.seq_length == 1:
        #     x = x[0]

        if self.transform is not None:
            x = self.transform(x)
        y = self.targets[true_index]
        return x, y

    def __len__(self):
        return self.targets.shape[0] * self.scale


def _csl_train_test_split(raw_data: dict, test_size=0.1):
    train_set = dict()
    test_set = dict()

    for i in raw_data.keys():
        length = len(raw_data[i])
        test_index = np.random.choice(length, int(length*test_size), replace=False)
        train_index = np.delete(np.arange(length), test_index)
        np_data = np.array(raw_data[i])
        test_set[i] = np_data[test_index].tolist()
        train_set[i] = np_data[train_index].tolist()

    return train_set, test_set


def _save_csl_to_csv(train_set: dict, test_set: dict, train_file_path, test_file_path):
    train_df = pd.DataFrame.from_dict(train_set)
    train_df.to_csv(train_file_path, index=False)

    test_df = pd.DataFrame.from_dict(test_set)
    test_df.to_csv(test_file_path, index=False)


def _load_csl_from_csv(train_file_path, test_file_path):
    train_df = pd.read_csv(train_file_path)
    train_set = train_df.to_dict(orient='list')

    test_df = pd.read_csv(test_file_path)
    test_set = test_df.to_dict(orient='list')

    return train_set, test_set


def _prepare_data(data_set: dict, gesture_list):
    """convert the data from raw data to numpy array
    """
    X = list()
    y = list()
    for i in range(len(gesture_list)):
        gesture = gesture_list[i]
        amount = len(data_set[gesture])
        X += data_set[gesture]
        y += [i for _ in range(amount)]

    y = np.asarray(y, dtype=np.int)
    return X, y


def _load_csl_all():
    subjects = ['subject1', 'subject2', 'subject3', 'subject4', 'subject5']
    csl_data = {}
    for s in subjects:
        subject_data = _load_csl_data(s)
        csl_data = _merge_data(csl_data, subject_data)

    return csl_data


def _load_csl_data(subject: str) -> dict:
    # get the parent dir path
    now_path = os.path.join(os.sep, *os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-2])
    data_path = os.path.join(now_path, 'data', 'csl', subject)
    session_list = os.listdir(data_path)
    subject_data = {}
    for session_folder in session_list:
        if not re.match(r'', session_folder, flags=0):
            # it is not session folders
            continue
        else:
            sess_path = os.path.join(data_path, session_folder)
            mat_files = os.listdir(sess_path)
            one_session_data = _read_csl_mat_files(sess_path, mat_files)
            subject_data = _merge_data(subject_data, one_session_data)

    return subject_data


def _read_csl_mat_files(path: str, mat_list: list) -> dict:
    mat_data = dict()

    for mat_name in mat_list:
        if not re.match(r'\S*[.]mat$', mat_name, flags=0):
            continue
        mat_path = os.path.join(path, mat_name)
        for i in range(10):
            file_path = mat_path + '{:d}'.format(i)
            if mat_name not in mat_data.keys():
                mat_data[mat_name] = list()
            mat_data[mat_name].append(file_path)
    return mat_data


def _merge_data(data_a: dict, data_b: dict) -> dict:
    all_gestures = set(data_a.keys()).union(set(data_b.keys()))
    emg_data = {i: [] for i in all_gestures}
    for g in all_gestures:
        if g in data_a.keys():
            emg_data[g] += data_a[g]
        if g in data_b.keys():
            emg_data[g] += data_b[g]

    return emg_data


if __name__ == '__main__':
    # 根据subject读取数据，证明针对一个人的预训练模型在其他subject的数据上表现未必足够好
    # 实验设计
    # 1. 在capg数据上预训练模型（或者载入训练好的模型）
    # 2. 在csl的部分数据上训练模型
    # 3. 用剩余subject的数据测试模型

    # # test pytorch data loader
    # train_data = CSLDataset(gesture=8, sequence_len=10, sequence_result=False,
    #                          frame_x=False, train=True, transform=None)
    # print(train_data.data.shape)
    # print(train_data.targets.shape)
    #
    # test_data = CSLDataset(gesture=8, sequence_len=10, sequence_result=False,
    #                         frame_x=False, train=False, transform=None)
    # print(test_data.data.shape)
    # print(test_data.targets.shape)
    test_data = CSLDataset()
    train_loader = DataLoader(test_data,
                              batch_size=1,
                              num_workers=4,
                              shuffle=True)

    for batch_idx, (data, target) in enumerate(train_loader):
        print(batch_idx)
        print(data.size())
        print(target)
        break

    print()
