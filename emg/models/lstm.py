# -*- coding: UTF-8 -*-
# lstm.py
# @Time     : 15/May/2019
# @Auther   : TENG HUO
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


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_num, dp):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layer_num,
            batch_first=True,
            dropout=dp,
            bidirectional=False
        )
        # self.bn1 = nn.BatchNorm1d(input_size, momentum=0.9)
        # self.bn2 = nn.BatchNorm1d(hidden_size, momentum=0.9)
        self.bn3 = nn.BatchNorm1d(hidden_size, momentum=0.9)
        # self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # for i in range(x.size(1)):
        #     x[:, i, :] = self.bn1(x[:, i, :])
        x, _ = self.rnn(x, None)   # None represents zero initial hidden state
        # choose r_out at the last time step
        x = x[:, -1, :]
        # x = self.bn2(x)
        # x = self.fc1(F.relu(x))
        x = self.bn3(x)
        x = self.fc2(F.relu(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return x


def main(train_args, TEST_MODE=False):
    if train_args['dataset'] == 'capg':
        args = {**train_args, **capg_args}
    else:
        args = {**train_args, **csl_args}
    all_gestures = list(range(args['gesture_num']))

    model = LSTM(args['input_size'], args['hidden_size'], len(all_gestures),
                 args['layer'], args['dropout'])
    name = args['name']
    sub_folder = args['sub_folder']

    # from emg.utils import config_tensorboard
    # tensorboard_cb = config_tensorboard(name, sub_folder, model, (1, 10, 128))

    from emg.utils.lr_scheduler import DecayLR
    lr_callback = DecayLR(start_lr=args['lr'], gamma=0.5, step_size=args['lr_step'])

    net = EMGClassifier(module=model,
                        model_name=name,
                        sub_folder=sub_folder,
                        hyperparamters=args,
                        optimizer=torch.optim.Adam,
                        gesture_list=all_gestures,
                        callbacks=[lr_callback])

    net = train(net, all_gestures)

    _ = test(net, all_gestures)

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
                                sequence_len=20,
                                gesture_list=gesture_indices,
                                train=True)
    else:
        train_set = CSLDataset(gesture=len(gesture_indices),
                               sequence_len=20,
                               train=True)
    net.dataset = train_set
    net.fit_with_dataset()
    return net


def test(net: EMGClassifier, gesture_indices: list):
    if net.hyperparamters['dataset'] == 'capg':
        test_set = CapgDataset(gestures_label_map=net.gesture_map,
                               sequence_len=20,
                               gesture_list=gesture_indices,
                               train=False)
    else:
        test_set = CSLDataset(gesture=len(gesture_indices),
                              sequence_len=20,
                              train=False)

    avg_score = net.test_model(gesture_indices, test_set)
    print('test accuracy: {:.4f}'.format(avg_score))
    return net


capg_args = {
    'input_size': 128,
    'hidden_size': 256,
    'seq_length': 20,
    'layer': 2,
    'dropout': 0.3
}

csl_args = {
    'input_size': 168,
    'hidden_size': 256,
    'seq_length': 20,
    'layer': 2,
    'dropout': 0.3
}


if __name__ == "__main__":
    test_args = {
        'model': 'lstm',
        'name': 'CSL-Test',
        'sub_folder': 'LSTM-test4',
        'dataset': 'csl',
        'gesture_num': 8,
        'epoch': 30,
        'train_batch_size': 256,
        'valid_batch_size': 1024,
        'lr': 0.0001,
        'lr_step': 5,
        'stop_patience': 5}

    print('test')
    main(test_args)

    # for i in [10, 15, 20, 30, 50]:
    #     hyperparameters['seq_length'] = int(i)
    #     main(test_args, TEST_MODE=False)

    # for i in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    #     hyperparameters['dropout'] = i
    #     test_args['name'] = 'lstm-dropout'
    #     test_args['sub_folder'] = 'dp-{}'.format(i)
    #     main(test_args, TEST_MODE=False)

    # for i in [1, 2, 3, 4]:
    #     hyperparameters['layer'] = i
    #     test_args['name'] = 'lstm-layer'
    #     test_args['sub_folder'] = 'layer-{}'.format(i)
    #     main(test_args, TEST_MODE=False)
