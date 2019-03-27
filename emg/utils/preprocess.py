# -*- coding: UTF-8 -*-
# preprocess.py
# @Time     : 27/Mar/2019
# @Auther   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
#

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

if __name__ == "__main__":
    # for test
    print('empty file')
