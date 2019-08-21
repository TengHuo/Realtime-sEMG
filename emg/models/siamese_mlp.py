# -*- coding: utf-8 -*-
# siamese_mlp.py
# @Time     : 03/Jul/2019
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


class SiameseMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SiameseMLP, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.pred = nn.Linear(hidden_size, 128)

    def embedding(self, x):
        x = self.bn1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        return self.pred(x)

    def forward(self, anchor, positive, negative):
        embedded_anchor = self.embedding(anchor)
        embedded_positive = self.embedding(positive)
        embedded_negative = self.embedding(negative)
        return embedded_anchor, embedded_positive, embedded_negative


hyperparameters = {
    'input_size': 128,
    'hidden_size': 256
}


def main(train_args):
    # 1. 设置好optimizer
    # 2. 定义好model
    args = {**train_args, **hyperparameters}
    model = SiameseMLP(args['input_size'], args['hidden_size'])
    name = args['name']
    sub_folder = args['sub_folder']

    # from emg.utils import config_tensorboard
    # tensorboard_cb = config_tensorboard(name, sub_folder, model, (1, 128))

    # from emg.utils.lr_scheduler import DecayLR
    # lr_callback = DecayLR(start_lr=args['lr'], gamma=0.5, step_size=args['lr_step'])

    net = SiameseEMG(module=model,
                     model_name=name,
                     sub_folder=sub_folder,
                     hyperparamters=args,
                     optimizer=torch.optim.Adam,
                     gesture_list=[],
                     callbacks=[])

    train(net)

    # test_set = CapgTriplet(gesture=args['gesture_num'],
    #                        sequence_len=1,
    #                        test_mode=TEST_MODE,
    #                        train=False)
    #
    # avg_score = net.test_model(test_set)
    # print('test accuracy: {:.4f}'.format(avg_score))


def train(net: SiameseEMG):
    gesture_list = list(range(8))
    train_set = CapgTriplet(gesture_list,
                            sequence_len=1,
                            frame_x=False,
                            train=True)
    net.dataset = train_set
    net.fit_with_dataset()
    return net


if __name__ == "__main__":
    test_args = {
        'model': 'siamese_mlp',
        'name': 'siamese-mlp',
        'sub_folder': 'test1',
        'epoch': 10,
        'train_batch_size': 512,
        'valid_batch_size': 2048,
        'lr': 0.001,
        'lr_step': 20}

    print('test')
    main(test_args)
