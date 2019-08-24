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
import h5py

import torch
from torch.utils.data import DataLoader, Dataset
from scipy.io import loadmat
import numpy as np

from multiprocessing import Pool
from emg.data_loader.preprocess import csl_preprocess


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

    def __init__(self, gesture, sequence_len, train,
                 frame_x=False, test_mode=False, transform=None):

        self.transform = transform
        self.train = train
        self.scale = 100
        self.seq_length = sequence_len
        self.frame_x = frame_x

        gestures = map(lambda i: 'gest{}'.format(i), range(gesture))  # from gest0 to gestX

        root_path = os.path.join(os.sep, *os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-2])
        processed_data_folder = os.path.join(root_path, 'data', 'csl-processed')
        if os.path.isdir(processed_data_folder):
            print('processed csl data exist, load {} data from the file'.format('train' if train else 'test'))
            print('it may take a few minutes...')
            data_set = _load_csl_from_h5(processed_data_folder, self.train, gestures)
        else:
            print('processed csl data not exist, create new h5 files')
            os.mkdir(processed_data_folder)
            csl_data = _load_csl_all()
            train_set, test_set = _csl_train_test_split(csl_data, test_size=0.1)
            train_data_path = os.path.join(processed_data_folder, 'train_')
            test_data_path = os.path.join(processed_data_folder, 'test_')
            data_set = _save_csl_to_h5(train_set, test_set, train_data_path, test_data_path, self.train, gestures)

        X, y = _prepare_data(data_set)
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
        x = self.data[true_index]
        start = np.random.randint(0, x.shape[0] - self.seq_length)
        end = start + self.seq_length
        x = x[start:end]
        if self.frame_x:
            if self.seq_length == 1:
                # used for 2d cnn
                x = x.reshape((1, 21, 8))
            else:
                # used for 3d cnn
                x = x.reshape((1, x.shape[0], 21, 8))
        elif self.seq_length == 1:
            x = np.squeeze(x)

        x = torch.from_numpy(x)
        x = x.float()
        if self.transform is not None:
            x = self.transform(x)
        y = self.targets[true_index]
        return x, y

    def __len__(self):
        return len(self.targets) * self.scale


def _csl_train_test_split(csl_data_dict: dict, test_size=0.1):
    train_set = dict()
    test_set = dict()

    for gesture in csl_data_dict.keys():
        data_list = csl_data_dict[gesture]
        length = len(data_list)
        test_index = np.random.choice(length, int(length*test_size), replace=False)
        train_set[gesture] = []
        test_set[gesture] = []
        for i in range(length):
            if i in test_index:
                test_set[gesture].append(data_list[i])
            else:
                train_set[gesture].append(data_list[i])

    return train_set, test_set


def _save_csl_to_h5(train_set: dict, test_set: dict, train_file_path, test_file_path,
                    train, gestures):
    _save_h5_file(train_set, train_file_path, 'train')
    _save_h5_file(test_set, test_file_path, 'test')
    if train:
        required_data = train_set
    else:
        required_data = test_set
    data_set = {}
    for g in gestures:
        data_set[g] = required_data[g]
    return data_set


def _save_h5_file(data_set, folder_path, grp_name):
    for gesture in data_set.keys():
        file_path = folder_path + '{}.h5'.format(gesture)
        with h5py.File(file_path, 'w') as h5_file:
            train_grp = h5_file.create_group(grp_name)
            for i, emg_data in enumerate(data_set[gesture]):
                train_grp.create_dataset(str(i), data=emg_data)


def _load_csl_from_h5(data_folder: str, train: bool, gestures):
    data_set = {}
    data_grp = 'train' if train else 'test'
    for g in gestures:
        file_path = os.path.join(data_folder, '{}_{}.h5'.format(data_grp, g))
        data_set[g] = _load_h5_file(file_path, data_grp)

    return data_set


def _load_h5_file(file_path, grp):
    data_list = []
    with h5py.File(file_path, 'r') as h5_file:
        data_grp = h5_file[grp]
        for i in data_grp.keys():
            data_list.append(data_grp[i][:])

    return data_list


def _prepare_data(data_set: dict):
    """convert the data from raw data to numpy array
    """
    X = list()
    y = list()
    for i, g in enumerate(data_set.keys()):
        data_list = data_set[g]
        amount = len(data_list)
        X += data_list
        y += [i for _ in range(amount)]

    return X, y


def _load_csl_all():
    subjects = ['subject1', 'subject2', 'subject3', 'subject4', 'subject5']
    csl_files = []
    for s in subjects:
        files_list = _load_csl_files(s)
        csl_files += files_list

    import time
    t1 = time.time()
    with Pool() as pool:
        csl_data = pool.map(_read_one_mat_file, csl_files)
    t2 = time.time()
    print('running time: ', int(t2 - t1))
    csl_data_dict = {}
    for gesture_data in csl_data:
        gesture = gesture_data[0]
        data_list = gesture_data[1]
        if gesture not in csl_data_dict.keys():
            csl_data_dict[gesture] = []
        csl_data_dict[gesture] += data_list

    return csl_data_dict


def _load_csl_files(subject: str) -> list:
    """load file name as a list"""
    # get the parent dir path
    now_path = os.path.join(os.sep, *os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-2])
    data_path = os.path.join(now_path, 'data', 'csl', subject)
    session_list = os.listdir(data_path)
    subject_files = []
    for session_folder in session_list:
        if not re.match(r'session[0-9]', session_folder, flags=0):
            # it is not a session folder
            continue
        else:
            sess_path = os.path.join(data_path, session_folder)
            mat_files = os.listdir(sess_path)
            one_session_files = _read_csl_mat_files(sess_path, mat_files)
            subject_files += one_session_files

    return subject_files


def _read_csl_mat_files(session_path: str, mat_list: list) -> list:
    mat_files = []

    for mat_name in mat_list:
        mat_path = os.path.join(session_path, mat_name)
        mat_name = mat_name[0:-4]  # extract name from gest1.mat to gest1
        mat_files.append((mat_name, mat_path))
    return mat_files


def _read_one_mat_file(file_tuple):
    file_name, file_path = file_tuple
    mat = loadmat(file_path)
    trials = mat['gestures']
    data_list = []
    for i in range(trials.shape[0]):
        one_trial = trials[i, 0]
        one_trial = csl_preprocess(one_trial)
        data_list.append(one_trial)

    res = (file_name, data_list)
    return res


if __name__ == '__main__':
    # test pytorch data loader
    csl_train_data = CSLDataset(gesture=8,
                                frame_x=True,
                                sequence_len=10,
                                train=True)
    print(len(csl_train_data.data))
    print(len(csl_train_data.targets))

    dataloader = DataLoader(csl_train_data, batch_size=512,
                            shuffle=True, num_workers=4)

    for i_batch, data in enumerate(dataloader):
        print(i_batch)
        print(data[0].size())
        print(data[1])
        break

    # csl_test_data = CSLDataset(gesture=8, sequence_len=10, train=False)
    # print(len(csl_test_data.data))
    # print(len(csl_test_data.targets))
