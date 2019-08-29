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
from emg.data_loader.capg_data import CapgDataset
from emg.data_loader.csl_data import CSLDataset


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


capg_args = {
    'input_size': 128,
    'hidden_size': 256
}

csl_args = {
    'input_size': 168,
    'hidden_size': 256
}


def main(train_args, TEST_MODE=False):
    if train_args['dataset'] == 'capg':
        args = {**train_args, **capg_args}
    else:
        args = {**train_args, **csl_args}

    if args['gesture_num'] == 8:
        all_gestures = list(range(8))
    elif args['gesture_num'] == 12:
        all_gestures = list(range(8, 20))
    else:
        all_gestures = list(range(args['gesture_num']))

    model = MLP(args['input_size'], args['hidden_size'], len(all_gestures))
    name = args['name']
    sub_folder = args['sub_folder']

    # from emg.utils import config_tensorboard
    # tensorboard_cb = config_tensorboard(name, sub_folder, model, (1, 128))

    from emg.utils.lr_scheduler import DecayLR
    lr_callback = DecayLR(start_lr=args['lr'], gamma=0.5, step_size=args['lr_step'])

    net = EMGClassifier(module=model,
                        model_name=name,
                        sub_folder=sub_folder,
                        hyperparamters=args,
                        optimizer=torch.optim.Adam,
                        gesture_list=all_gestures,
                        callbacks=[lr_callback])

    if not TEST_MODE:
        net = train(net, all_gestures)

    confusion_matrx = test(net, all_gestures)
    return confusion_matrx
    # test_gestures = all_gestures[0:1]
    # net = test(net, test_gestures)
    #
    # test_gestures = all_gestures[1:2]
    # net = test(net, test_gestures)
    #
    # test_gestures = all_gestures[2:3]
    # net = test(net, test_gestures)


def train(net: EMGClassifier, gesture_indices: list):
    if net.hyperparamters['dataset'] == 'capg':
        train_set = CapgDataset(gestures_label_map=net.gesture_map,
                                sequence_len=1,
                                gesture_list=gesture_indices,
                                train=True)
    else:
        train_set = CSLDataset(gesture=8,
                               sequence_len=1,
                               train=True)
    net.dataset = train_set
    net.fit_with_dataset()
    return net


def test(net: EMGClassifier, gesture_indices: list):
    if net.hyperparamters['dataset'] == 'capg':
        test_set = CapgDataset(gestures_label_map=net.gesture_map,
                               sequence_len=1,
                               gesture_list=gesture_indices,
                               train=False)
    else:
        test_set = CSLDataset(gesture=8,
                              sequence_len=1,
                              train=False)

    avg_score, matrix = net.test_model(gesture_indices, test_set)
    print('test accuracy: {:.4f}'.format(avg_score))
    return matrix


if __name__ == "__main__":
    test_args = {
        'model': 'mlp',
        'name': 'MLP-Unit-Test',
        'sub_folder': 'size-512',
        'dataset': 'csl',
        'gesture_num': 8,
        'epoch': 200,
        'train_batch_size': 512,
        'valid_batch_size': 2048,
        'lr': 0.001,
        'lr_step': 40}

    print('test')
    # default_name = test_argsg['model'] + '-{}'.format(test_args['suffix'])
    # test_args['name'] = default_name
    # test_args['name'] = '8Gesture_Compare'
    # test_args['name'] = '12Gesture_Compare'

    main(test_args)
