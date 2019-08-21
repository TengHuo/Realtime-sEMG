# -*- coding: utf-8 -*-
# siamese_lstm.py
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


class SiameseLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_num, dp):
        super(SiameseLSTM, self).__init__()

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
        self.output = nn.Linear(hidden_size, 128)

    def embedding(self, x):
        x, _ = self.rnn(x, None)
        x = x[:, -1, :]
        x = self.bn3(x)
        output = self.output(F.relu(x))
        return output

    def forward(self, anchor, positive, negative):
        embedded_anchor = self.embedding(anchor)
        embedded_positive = self.embedding(positive)
        embedded_negative = self.embedding(negative)
        return embedded_anchor, embedded_positive, embedded_negative


def main(train_args):
    # 1. 设置好optimizer
    # 2. 定义好model
    args = {**train_args, **hyperparameters}
    all_gestures = list(range(8))

    model = SiameseLSTM(args['input_size'], args['hidden_size'], len(all_gestures),
                        args['layer'], args['dropout'])
    name = args['name']
    sub_folder = args['sub_folder']

    # from emg.utils import config_tensorboard
    # tensorboard_cb = config_tensorboard(name, sub_folder, model, (1, 10, 128))
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

    # _ = test(net, all_gestures)

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
                            sequence_len=20,
                            frame_x=False,
                            train=True)
    net.dataset = train_set
    net.fit_with_dataset()
    return net


# def test(net: SiameseEMG, gesture_indices: list):
#     gesture_list = list(range(8))
#     test_set = CapgTriplet(gesture_list,
#                             sequence_len=20,
#                             frame_x=False,
#                             train=False)
#
#     avg_score = net.test_model(gesture_indices, test_set)
#     print('test accuracy: {:.4f}'.format(avg_score))
#     return net


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
        'name': 'siamese-lstm',
        'sub_folder': 'test1',
        'epoch': 10,
        'train_batch_size': 256,
        'valid_batch_size': 1024,
        'lr': 0.001,
        'lr_step': 50}

    print('test')
    # default_name = test_args['model'] + '-{}'.format(test_args['suffix'])
    # test_args['name'] = default_name

    # test_args['name'] = '8Gesture_Compare'
    # test_args['name'] = '12Gesture_Compare'
    # test_args['name'] = '20Gesture_Compare'
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

