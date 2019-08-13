# -*- coding: utf-8 -*-
# siamese_cnn.py
# @Time     : 27/Jul/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
#


import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from emg.models.siamese import SiameseEMG
from emg.data_loader.capg_triplet import CapgTriplet


class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size ** 2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


class SiameseCNN(nn.Module):
    def __init__(self, gesture_num):
        super(SiameseCNN, self).__init__()
        self.cov = nn.Sequential(
            nn.BatchNorm2d(1, momentum=0.9),
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(True),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(True),

            LocallyConnected2d(64, 64, (16, 8), 1, 1, bias=False),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(True),

            LocallyConnected2d(64, 64, (16, 8), 1, 1, bias=False),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(True),
            nn.Dropout(0.5),
        )
        self.flat = nn.Sequential(
            nn.Linear(8192, 512),
            nn.BatchNorm1d(512, momentum=0.9),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512, momentum=0.9),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128, momentum=0.9),
            nn.ReLU(True)
        )
        self.output = nn.Linear(128, 128)

    def embedding(self, x):
        x = self.cov(x)
        x = x.view(x.size(0), -1)
        x = self.flat(x)
        return self.output(x)

    def forward(self, anchor, positive, negative):
        embedded_anchor = self.embedding(anchor)
        embedded_positive = self.embedding(positive)
        embedded_negative = self.embedding(negative)
        return embedded_anchor, embedded_positive, embedded_negative


def main(train_args, TEST_MODE=False):
    args = train_args
    all_gestures = list(range(20))

    model = SiameseCNN(len(all_gestures))
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
                            sequence_len=1,
                            frame_x=True,
                            train=True)
    net.dataset = train_set
    net.fit_with_dataset()
    return net


# def test(net: SiameseEMG):
#     gesture_list = list(range(8))
#     test_set = CapgTriplet(gesture_list,
#                            sequence_len=1,
#                            frame_x=True,
#                            train=False)
#
#     avg_score = net.test_model(gesture_indices, test_set)
#     print('test accuracy: {:.4f}'.format(avg_score))
#     return net


if __name__ == "__main__":
    test_args = {
        'model': 'cnn',
        'name': 'siamese-cnn',
        'sub_folder': 'test1',
        'epoch': 30,
        'train_batch_size': 128,
        'valid_batch_size': 1024,
        'lr': 0.001,
        'lr_step': 50}

    print('test')
    # default_name = test_args['model'] + '-{}'.format(test_args['suffix'])
    # test_args['name'] = '8Gesture_Compare'
    # test_args['name'] = '12Gesture_Compare'
    main(test_args, TEST_MODE=False)
