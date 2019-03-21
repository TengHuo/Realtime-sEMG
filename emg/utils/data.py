import numpy as np
import os
from scipy.io import loadmat
import re
from enum import Enum, unique

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

def load_capg_data(db_name, mode=LoadMode.sequence):
    # get the parent dir path
    now_path = os.path.join(os.sep, *os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-2])
    data_path = os.path.join(now_path, 'data', 'capg', db_name.value)
    mat_list = os.listdir(data_path)
    x, y, category = _read_capg_mat_files(data_path, mat_list)
    
    if mode == LoadMode.sequence:
        # default shape (None, 1000, 128): sequence mode
        pass
    elif mode == LoadMode.sequence_frame:
        x = x.reshape(x.shape[0], x.shape[1], 16, 8, 1)
    elif mode == LoadMode.flat:
        # reshape to (None, 128)
        x = x.reshape(x.shape[0]*1000, 128)
        y = y.reshape(y.shape[0], 1) * np.ones((1, 1000))
        y = y.flatten()
    elif mode == LoadMode.flat_frame:
        # reshape to (None, 16, 8)
        x = x.reshape(x.shape[0]*1000, 16, 8, 1)
        y = y.reshape(y.shape[0], 1) * np.ones((1, 1000))
        y = y.flatten()

    return x, y, category

def _read_capg_mat_files(path, mat_list):
    x = list()
    y = list()
    gesture_set = set()
    for mat_name in mat_list:
        if not re.match(r'\S*[.]mat$', mat_name, flags=0):
            continue
        mat_path = os.path.join(path, mat_name)
        mat = loadmat(mat_path)
        gesture_index = mat['gesture'][0][0]
        if gesture_index in [100, 101]:
            continue
        else:
            gesture_set.add(gesture_index)
        
        x.append(mat['data'])
        y.append(gesture_index)
    return np.asarray(x), np.asarray(y), len(gesture_set)


if __name__ == '__main__':
    # For test
    seq_data_dim = {
        'dba': (1440, 1000, 128),
        'dbb': (1600, 1000, 128),
        'dbc': (1200, 1000, 128)
    }
    for db_name in seq_data_dim.keys():
        x, y, c = load_capg_data(db_name=CapgDBName[db_name])
        assert x.shape == seq_data_dim[db_name]

    flat_data_dim = {
        'dba': (1440000, 128),
        'dbb': (1600000, 128),
        'dbc': (1200000, 128)
    }
    for db_name in flat_data_dim.keys():
        x, y, c = load_capg_data(db_name=CapgDBName[db_name], mode=LoadMode.flat)
        print(x.shape)
        assert x.shape == flat_data_dim[db_name]







# import numpy as np
# from scipy.io import loadmat
# import os
# import re
# import h5py

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

#     # TODO:
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

#     # TODO:
#     def sampling(self, data, percentage):
#         pass



# # TODO: 添加单元测试
