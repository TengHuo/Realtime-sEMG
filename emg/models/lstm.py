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

from sklearn.model_selection import GridSearchCV

from emg.models.base import EMGClassifier
from emg.utils import config_tensorboard
from emg.data_loader.capg_data import CapgDataset


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
    # 1. 设置好optimizer
    # 2. 定义好model
    args = {**train_args, **hyperparameters}
    all_gestures = list(range(0, args['gesture_num']))

    model = LSTM(args['input_size'], args['hidden_size'], len(all_gestures),
                 args['layer'], args['dropout'])
    name = args['name']
    sub_folder = args['sub_folder']

    tensorboard_cb = config_tensorboard(name, sub_folder, model, (1, 10, 128))

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
    train_set = CapgDataset(gestures_label_map=net.gesture_map,
                            sequence_len=20,
                            gesture_list=gesture_indices,
                            train=True)
    net.dataset = train_set
    net.fit_with_dataset()
    return net


def test(net: EMGClassifier, gesture_indices: list):
    test_set = CapgDataset(gestures_label_map=net.gesture_map,
                           sequence_len=20,
                           gesture_list=gesture_indices,
                           train=False)

    avg_score = net.test_model(gesture_indices, test_set)
    print('test accuracy: {:.4f}'.format(avg_score))
    return net


hyperparameters = {
    'input_size': 128,
    'hidden_size': 256,
    'seq_length': 20,
    'layer': 2,
    'dropout': 0.3
}


if __name__ == "__main__":
    test_args = {
        'model': 'lstm',
        'suffix': 'test',
        'sub_folder': 'test1',
        'epoch': 1,
        'train_batch_size': 256,
        'valid_batch_size': 1024,
        'lr': 0.001,
        'lr_step': 50}

    print('test')
    default_name = test_args['model'] + '-{}'.format(test_args['suffix'])
    test_args['name'] = default_name
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
