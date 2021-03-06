# -*- coding: utf-8 -*-
# c3d64.py
# @Time     : 16/Jul/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
#


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from emg.models.base import EMGClassifier
from emg.data_loader.capg_data import CapgDataset


class C3D64(nn.Module):
    def __init__(self, output_size):
        super(C3D64, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(1, 1, 1))

        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 128)
        self.fc8 = nn.Linear(128, output_size)

    def forward(self, x):
        # print('SIZE1:')
        # print(x.size())
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = self.pool3(x)
        # print('SIZE2:')
        # print(x.size())

        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv4b(x))
        x = self.pool4(x)
        # print('SIZE3:')
        # print(x.size())

        # x = F.relu(self.conv5a(x))
        # x = F.relu(self.conv5b(x))
        # x = self.pool5(x)

        x = x.view(-1, 512)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, p=0.2)

        x = F.relu(self.fc7(x))
        x = F.dropout(x, p=0.2)
        output = self.fc8(x)
        return output


def main(train_args, TEST_MODE=False):
    args = train_args
    all_gestures = list(range(0, 8))

    model = C3D64(len(all_gestures))
    name = args['name']
    sub_folder = args['sub_folder']

    # from emg.utils import config_tensorboard
    # tensorboard_cb = config_tensorboard(name, sub_folder)
    #
    # from emg.utils.lr_scheduler import DecayLR
    # lr_callback = DecayLR(start_lr=args['lr'], gamma=0.5, step_size=args['lr_step'])

    net = EMGClassifier(module=model,
                        model_name=name,
                        sub_folder=sub_folder,
                        hyperparamters=args,
                        optimizer=torch.optim.Adam,
                        gesture_list=all_gestures,
                        callbacks=[])

    net = train(net, all_gestures)

    net = test(net, all_gestures)

    # test_gestures = all_gestures[0:1]
    # net = test(net, test_gestures)
    #
    # test_gestures = all_gestures[1:2]
    # net = test(net, test_gestures)
    #
    # test_gestures = all_gestures[2:3]
    # net = test(net, test_gestures)


def train(net: EMGClassifier, gesture_indices: list):
    transforms_list = transforms.Compose([
        # transforms.Resize((32, 32), interpolation=2),
        transforms.ToTensor(),
    ])

    train_set = CapgDataset(gestures_label_map=net.gesture_map,
                            sequence_len=10,
                            frame_x=True,
                            gesture_list=gesture_indices,
                            train=True,
                            transform=transforms_list)
    net.dataset = train_set
    net.fit_with_dataset()
    return net


def test(net: EMGClassifier, gesture_indices: list):
    transforms_list = transforms.Compose([
        # transforms.Resize((32, 32), interpolation=2),
        transforms.ToTensor(),
    ])

    test_set = CapgDataset(gestures_label_map=net.gesture_map,
                           sequence_len=10,
                           frame_x=True,
                           gesture_list=gesture_indices,
                           train=False,
                           transform=transforms_list)

    avg_score = net.test_model(gesture_indices, test_set)
    print('test accuracy: {:.4f}'.format(avg_score))
    return net


if __name__ == "__main__":
    test_args = {
        'model': 'c3d64',
        'name': 'c3d-upscale',
        'sub_folder': 'size24-test2',
        'epoch': 60,
        'train_batch_size': 128,
        'valid_batch_size': 512,
        'lr': 0.001,
        'lr_step': 5}

    print('test')
    main(test_args)
