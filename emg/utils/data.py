# -*- coding: UTF-8 -*-
# data.py
# @Time     : 22/Mar/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
#

import numpy as np
import os
from scipy.io import loadmat
import re
from enum import Enum, unique
from tensorflow.keras import utils
import h5py

# TODO: 1. implement a csl data loader
# TODO: 2. merge capg and csl data loader, train models with all data
# TODO: 3. implement a data generator for generating triplet pairs

@unique
class CapgDBName(Enum):
    # gesture index: {1, 2, 3, 4, 5, 6, 7, 8, 100, 101}
    dba = 'dba'
    # gesture index: {1, 2, 3, 4, 5, 6, 7, 8, 100, 101}
    dbb = 'dbb'
    # gesture index: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 100, 101}
    dbc = 'dbc'


LoadMode = Enum('LoadMode', ('flat', 'flat_frame', 'sequence', 'sequence_frame'))


def load_capg_all():
    # load three databases of capg    
    dba = _load_capg_data(CapgDBName.dba)
    dbb = _load_capg_data(CapgDBName.dbb)
    dbc = _load_capg_data(CapgDBName.dbc)

    capg = {gesture_index: None for gesture_index in range(20)}
    for i in dba.keys():
        capg_index = i - 1
        capg[capg_index] = np.asarray(dba[i] + dbb[i])

    for i in dbc.keys():
        capg_index = i + 7
        capg[capg_index] = np.asarray(dbc[i])

    return capg


def capg_train_test_split(data, test_size=0.1):
    train = dict()
    test = dict()

    for i in data.keys():
        shape = data[i].shape
        test_index = np.random.choice(shape[0], int(shape[0]*test_size), replace=False)
        train_index = np.delete(np.arange(shape[0]), test_index)
        test[i] = data[i][test_index]
        train[i] = data[i][train_index]

    return train, test


def prepare_data(data, required_gestures=8, mode=LoadMode.sequence):
    '''split train and test dataset
    '''
    required_range = range(required_gestures)
    X = list()
    y = list()
    for i in required_range:
        amount = data[i].shape[0]
        X.append(data[i])
        y += [i for _ in range(amount)]
    X = np.concatenate(X, axis=0)
    y = np.asarray(y)

    original_shape = X.shape
    if mode == LoadMode.sequence:
        # default shape (None, 1000, 128): sequence mode
        pass
    elif mode == LoadMode.sequence_frame:
        X = X.reshape(original_shape[0], 1000, 16, 8, 1)
    elif mode == LoadMode.flat:
        # reshape to (None, 128)
        X = X.reshape(original_shape[0]*1000, 128)
        y = y.reshape((y.shape[0], 1)) * np.ones((1, 1000))
        y = y.flatten()
    elif mode == LoadMode.flat_frame:
        # reshape to (None, 16, 8)
        X = X.reshape(original_shape[0]*1000, 16, 8, 1)
        y = y.reshape((y.shape[0], 1)) * np.ones((1, 1000))
        y = y.flatten()

    y = utils.to_categorical(y, required_gestures)
    return X, y


def load_capg_from_h5(file_name):
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


def save_capg_to_h5(train, test, file_name):
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
    data_path = os.path.join(now_path, 'data', 'capg', db_name.value)
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
