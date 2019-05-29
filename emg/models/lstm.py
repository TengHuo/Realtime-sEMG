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
from emg.models.torch_model import start_train


hyperparameters = {
    'input_size': (128,),
    'hidden_size': 256,
    'seq_length': 10,
    'seq_result': False,
    'frame_input': False
}


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.rnn = nn.GRU(
            input_size=input_size[0],
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        # self.bn1 = nn.BatchNorm1d(input_size[0], momentum=0.9)
        self.bn2 = nn.BatchNorm1d(hidden_size, momentum=0.9)
        self.bn3 = nn.BatchNorm1d(hidden_size, momentum=0.9)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # for i in range(x.size(1)):
        #     x[:, i, :] = self.bn1(x[:, i, :])
        x, _ = self.rnn(x, None)   # None represents zero initial hidden state
        # choose r_out at the last time step
        x = x[:, -1, :]
        x = self.bn2(x)
        x = self.fc1(F.relu(x))
        x = self.bn3(x)
        x = self.fc2(F.relu(x))
        return x


def main(train_args, test_mode=False):
    # 1. 设置好optimizer
    # 2. 定义好model
    args = {**train_args, **hyperparameters}
    model = LSTM(args['input_size'], args['hidden_size'], args['gesture_num'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])

    start_train(args, model, optimizer, test_mode)


if __name__ == "__main__":
    test_args = {
        'model': 'lstm',
        'gesture_num': 8,
        'lr': 0.001,
        'epoch': 10,
        'train_batch_size': 256,
        'val_batch_size': 1024,
        'stop_patience': 7,
        'log_interval': 100,
        'load_model': False
    }

    main(test_args, test_mode=False)
