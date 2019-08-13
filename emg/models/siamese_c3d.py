# -*- coding: utf-8 -*-
# siamese_c3d.py
# @Time     : 27/Jul/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
#


import torch
import torch.nn as nn
import torch.nn.functional as F

from emg.models.siamese import SiameseEMG
from emg.data_loader.capg_triplet import CapgTriplet


class SiameseC3D(nn.Module):
    def __init__(self, output_size):
        super(SiameseC3D, self).__init__()
        # self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        #
        # self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv4a = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(3072, 512)
        self.fc7 = nn.Linear(512, 512)
        self.output = nn.Linear(512, 128)

    def embedding(self, x):
        # x = F.relu(self.conv1(x))
        # x = self.pool1(x)
        #
        # x = F.relu(self.conv2(x))
        # x = self.pool2(x)

        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = self.pool3(x)

        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv4b(x))
        x = self.pool4(x)

        x = F.relu(self.conv5a(x))
        x = F.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.view(-1, 3072)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, p=0.2)

        x = F.relu(self.fc7(x))
        x = F.dropout(x, p=0.2)
        output = self.output(x)
        return output

    def forward(self, anchor, positive, negative):
        embedded_anchor = self.embedding(anchor)
        embedded_positive = self.embedding(positive)
        embedded_negative = self.embedding(negative)
        return embedded_anchor, embedded_positive, embedded_negative


def main(train_args, TEST_MODE=False):
    # 1. 设置好optimizer
    # 2. 定义好model
    args = train_args
    all_gestures = list(range(8))

    model = SiameseC3D(len(all_gestures))
    name = args['name']
    sub_folder = args['sub_folder']

    # from emg.utils import config_tensorboard
    # tensorboard_cb = config_tensorboard(name, sub_folder)
    #
    # from emg.utils.lr_scheduler import DecayLR
    # lr_callback = DecayLR(start_lr=args['lr'], gamma=0.5, step_size=args['lr_step'])

    net = SiameseEMG(module=model,
                     model_name=name,
                     sub_folder=sub_folder,
                     hyperparamters=args,
                     optimizer=torch.optim.Adam,
                     gesture_list=[],
                     callbacks=[])

    net = train(net)

    # net = test(net, all_gestures)

    # test_gestures = all_gestures[0:1]
    # net = test(net, test_gestures)
    #
    # test_gestures = all_gestures[1:2]
    # net = test(net, test_gestures)
    #
    # test_gestures = all_gestures[2:3]
    # net = test(net, test_gestures)


def train(net: SiameseEMG):
    gesture_list = list(range(8))
    train_set = CapgTriplet(gesture_list,
                            sequence_len=10,
                            frame_x=True,
                            train=True)
    net.dataset = train_set
    net.fit_with_dataset()
    return net


# def test(net: SiameseEMG):
#     gesture_list = list(range(8))
#     test_set = CapgTriplet(gesture_list,
#                            sequence_len=10,
#                            frame_x=True,
#                            train=False)
#
#     avg_score = net.test_model(gesture_indices, test_set)
#     print('test accuracy: {:.4f}'.format(avg_score))
#     return net


if __name__ == "__main__":
    test_args = {
        'model': 'c3d',
        'name': 'siamese-c3d',
        'sub_folder': 'test1',
        'epoch': 1,
        'train_batch_size': 256,
        'valid_batch_size': 1024,
        'lr': 0.001,
        'lr_step': 5}

    print('test')
    # default_name = test_args['model'] + '-{}'.format(test_args['suffix'])
    # test_args['name'] = default_name
    # test_args['name'] = '8Gesture_Compare'
    # test_args['name'] = '12Gesture_Compare'
    # test_args['name'] = '20Gesture_Compare'
    main(test_args)

