# -*- coding: UTF-8 -*-
# mlp.py
# @Time     : 24/May/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
#


import torch
import torch.nn as nn
import torch.nn.functional as F

from emg.models.base import EMGClassifier
from emg.utils import config_tensorboard
from emg.data_loader.capg_data import CapgDataset


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # an affine operation: y = Wx + b
        self.bn1 = nn.BatchNorm1d(input_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.bn1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


hyperparameters = {
    'input_size': 128,
    'hidden_size': 256
}


def main(train_args, TEST_MODE=False):
    # 1. 设置好optimizer
    # 2. 定义好model
    args = {**train_args, **hyperparameters}
    all_gestures = [2, 3, 4, 5, 6, 7, 8, 9]
    model = MLP(args['input_size'], args['hidden_size'], len(all_gestures))
    name = args['name']
    sub_folder = args['sub_folder']

    tensorboard_cb = config_tensorboard(name, sub_folder, model, (1, 128))

    from emg.utils.lr_scheduler import DecayLR
    lr_callback = DecayLR(start_lr=args['lr'], gamma=0.5, step_size=args['lr_step'])

    net = EMGClassifier(module=model,
                        model_name=name,
                        sub_folder=sub_folder,
                        hyperparamters=args,
                        optimizer=torch.optim.Adam,
                        gesture_list=all_gestures,
                        callbacks=[tensorboard_cb, lr_callback])

    net = train(net, all_gestures)

    net = test(net, all_gestures)

    test_gestures = all_gestures[0:1]
    net = test(net, test_gestures)

    test_gestures = all_gestures[1:2]
    net = test(net, test_gestures)

    test_gestures = all_gestures[2:3]
    net = test(net, test_gestures)


def train(net: EMGClassifier, gesture_indices: list):
    train_set = CapgDataset(gestures_label_map=net.gesture_map,
                            sequence_len=1,
                            gesture_list=gesture_indices,
                            train=True)
    net.dataset = train_set
    net.fit_with_dataset()
    return net


def test(net: EMGClassifier, gesture_indices: list):
    test_set = CapgDataset(gestures_label_map=net.gesture_map,
                           sequence_len=1,
                           gesture_list=gesture_indices,
                           train=False)

    avg_score = net.test_model(test_set)
    print('test accuracy: {:.4f}'.format(avg_score))
    return net


if __name__ == "__main__":
    test_args = {
        'model': 'mlp',
        'suffix': 'test-test',
        'sub_folder': 'test2',
        # 'gesture_num': 8,
        'epoch': 60,
        'train_batch_size': 512,
        'valid_batch_size': 2048,
        'lr': 0.001,
        'lr_step': 50}

    print('test')
    default_name = test_args['model'] + '-{}'.format(test_args['suffix'])
    test_args['name'] = default_name
    main(test_args, TEST_MODE=False)
