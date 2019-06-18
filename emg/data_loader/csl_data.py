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
from scipy.ndimage.filters import median_filter
from scipy.signal import butter, lfilter
from multiprocessing import Pool


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


def butter_bandpass_filter(emg_data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='bandpass')
    y = lfilter(b, a, emg_data)
    return y


def continuous_segments(label):
    label = np.asarray(label)

    if not len(label):
        return

    breaks = list(np.where(label[:-1] != label[1:])[0] + 1)
    for begin, end in zip([0] + breaks, breaks + [len(label)]):
        assert begin < end
        yield begin, end


def csl_cut(emg_data, framerate):
    window = int(np.round(150 * framerate / 2048))
    emg_data = emg_data[:len(emg_data) // window * window].reshape(-1, 150, emg_data.shape[1])
    rms = np.sqrt(np.mean(np.square(emg_data), axis=1))
    rms = [median_filter(image, 3).ravel() for image in rms.reshape(-1, 24, 7)]
    rms = np.mean(rms, axis=1)
    threshold = np.mean(rms)
    mask = rms > threshold
    for i in range(1, len(mask) - 1):
        if not mask[i] and mask[i - 1] and mask[i + 1]:
            mask[i] = True
    begin, end = max(continuous_segments(mask),
                     key=lambda s: (mask[s[0]], s[1] - s[0]))
    return begin * window, end * window


def csl_preprocess(trial):
    trial = np.delete(trial, np.s_[7:192:8], 0)
    trial = trial.T

    # bandpass
    trial = np.transpose([butter_bandpass_filter(ch, 20, 400, 2048, 4) for ch in trial.T])
    # cut
    begin, end = csl_cut(trial, 2048)
    # print(begin, end)
    trial = trial[begin:end]
    # median filter
    trial = np.array([median_filter(image, 3).ravel() for image in trial.reshape(-1, 24, 7)])
    return trial


class CSLDataset(Dataset):
    """An abstract class representing a Dataset.
    """

    def __init__(self, sequence_len, train=True, test_mode=False, transform=None):

        self.transform = transform
        self.train = train
        self.scale = 100
        self.seq_length = sequence_len

        root_path = os.path.join(os.sep, *os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-2])
        processed_data = os.path.join(root_path, 'data', 'csl-processed')
        train_data_path = os.path.join(processed_data, 'train.csv')
        test_data_path = os.path.join(processed_data, 'test.csv')
        if os.path.isfile(train_data_path) and os.path.isfile(test_data_path):
            print('processed csl data exist, load {} data from the file'.format('train' if train else 'test'))
            train_set, test_set = _load_csl_from_csv(train_data_path, test_data_path)
        else:
            print('processed csl data not exist, create new h5 files')
            os.mkdir(processed_data)
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
        x = mat['gestures'][x_index][0]
        x = csl_preprocess(x)
        start = np.random.randint(0, x.shape[1] - self.seq_length)
        end = start + self.seq_length
        x = x[start:end]

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
        # TODO: 需要修改
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
        if not re.match(r'', session_folder, flags=0):
            # it is not session folders
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
        mat_files.append((mat_name, mat_path))
    return mat_files


def _merge_data(data_a: dict, data_b: dict) -> dict:
    all_gestures = set(data_a.keys()).union(set(data_b.keys()))
    emg_data = {i: [] for i in all_gestures}
    for g in all_gestures:
        if g in data_a.keys():
            emg_data[g] += data_a[g]
        if g in data_b.keys():
            emg_data[g] += data_b[g]

    return emg_data


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
    # TODO: 需要写一个将raw data预处理一遍的函数
    # 将数据处理为和capg一样的格式存到h5文件中
    # {gesture1: [ndarray], gesture2: [ndarray]}
    # csl_tra

    test = _load_csl_all()
    print()

