# -*- coding: UTF-8 -*-
# data.py
# @Time     : 22/Mar/2019
# @Auther   : TENG HUO
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

# TODO: 将函数封装到一个类中
class CapgData(object):
    pass

def load_capg_all(mode=LoadMode.sequence):

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

    for i in capg.keys():
        original_shape = capg[i].shape
        if mode == LoadMode.sequence:
            # default shape (None, 1000, 128): sequence mode
            continue
        elif mode == LoadMode.sequence_frame:
            capg[i] = capg[i].reshape(original_shape[0], 1000, 16, 8, 1)
        elif mode == LoadMode.flat:
            # reshape to (None, 128)
            capg[i] = capg[i].reshape(original_shape[0]*1000, 128)
        elif mode == LoadMode.flat_frame:
            # reshape to (None, 16, 8)
            capg[i] = capg[i].reshape(original_shape[0]*1000, 16, 8, 1)
    return capg

def capg_split_train_test(data, test_size=0.2, random_state=None):
    train = {gesture_index: None for gesture_index in range(20)}
    test = {gesture_index: None for gesture_index in range(20)}

    for i in data.keys():
        shape = data[i].shape
        test_index = np.random.choice(shape[0], int(shape[0]*test_size))
        test[i] = data[i][test_index]
        train[i] = np.delete(data[i], test_index, axis=0)

    return train, test

def prepare_data(data, required_gestures=8):
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
    return X, y

def load_capg_from_h5(file_name):
    with h5py.File(file_name, 'r') as data_file:
        train = np.array(data_file['train'])
        test = np.array(data_file['test'])
    return train, test

def save_capg_to_h5(train, test, file_name):
    with h5py.File(file_name, 'w') as data_file:
        data_file.create_dataset('train', data=train, dtype=np.float)
        data_file.create_dataset('test', data=test, dtype=np.float)
        
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


if __name__ == '__main__':
    # For test
    test = load_capg_all(LoadMode.sequence)
    for i in test.keys():
        print(test[i].shape)

    test = load_capg_all(LoadMode.sequence_frame)
    for i in test.keys():
        print(test[i].shape)

    test = load_capg_all(LoadMode.flat_frame)
    for i in test.keys():
        print(test[i].shape)

    now_path = os.path.join(os.sep, *os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-2])
    h5_file_path = os.path.join(now_path, 'cache', 'test.h5')

    if not os.path.isfile(h5_file_path):
        test = load_capg_all(LoadMode.flat)
        train, test = capg_split_train_test(test, test_size=0.1)
        save_capg_to_h5(train, test, h5_file_path)
    else:
        train, test = load_capg_from_h5(h5_file_path)
    
    x_train, y_train = prepare_data(train)
    x_test, y_test = prepare_data(test)
    print(x_train.shape)
    print(y_train.shape)
    print()
    print(x_test)
    print(y_test)


# TODO: add data preprocessing code
# 例如转为RGB，low pass filter等等

# class Preprocess(object):
#     def emg_to_frames(self, emg_data, normal_value=2.5, scale=10000, image_size=(8, 16), rgb=True):
#         # normalization the data to [0, 1]
#         emg_data = (emg_data * scale + normal_value*scale) / (normal_value*2*scale)
#         if rgb:
#             frames_size = (emg_data.shape[0], image_size[0], image_size[1], 3)
#         else:
#             frames_size = (emg_data.shape[0], image_size[0], image_size[1], 1)
#         emg_frames = np.empty(frames_size)
#         for i in range(emg_data.shape[0]):
#             frame = np.reshape(emg_data[i], image_size)
#             if rgb:
#                 # replicate three times as RGB channel
#                 frame_3d = frame[:,:,None] * np.ones(3, dtype=int)[None,None,:]
#                 emg_frames[i] = frame_3d
#             else:
#                 frame_1d = frame
#                 emg_frames[i] = frame_1d
#         return emg_frames

#     def read_row_data(self, db_name, input_dir):
#         # input: a directory
#         # output: gestures

#         # DB_NAME = db_name
#         # INPUT_DIR = '../data/capg_raw/'

#         gestures = dict()
#         for root, directories, files in os.walk(input_dir):
#             if not re.search(db_name, root):
#                 continue
#             print('root:{}, directories:{}, files: {}'.format(root, len(directories), len(files)))
#             for file in files:
#                 gesture_num = file[4:7]
#                 if gesture_num not in gestures.keys():
#                     gestures[gesture_num] = list()
#                 file_loc = os.path.join(root, file)
#                 mat = loadmat(file_loc)
#                 gestures[gesture_num].append(convert_emg(mat))

#         print(gestures.keys())

#         return gestures
