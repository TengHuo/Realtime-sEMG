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
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size[0],
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            dropout=0.2,
        )
        # self.bn1 = nn.BatchNorm1d(input_size[0], momentum=0.9)
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
        x = F.dropout(x, p=0.2, training=self.training)
        return x


def main(train_args, TEST_MODE=False):
    # 1. 设置好optimizer
    # 2. 定义好model
    args = {**train_args, **hyperparameters}
    # TODO: 修改代码，添加Grid Search
    model = LSTM(args['input_size'], args['hidden_size'], args['gesture_num'])
    name = args['model'] + '-' + str(args['gesture_num'])
    sub_folder = 'length-{}'.format(args['seq_length'])
    tensorboard_cb = config_tensorboard(name, sub_folder, model, (1, 10, 128))

    # from emg.utils.lr_scheduler import DecayLR
    # lr_callback = DecayLR(start_lr=0.001, gamma=0.1, step_size=12)

    train_set = CapgDataset(gesture=args['gesture_num'],
                            sequence_len=10,
                            sequence_result=False,
                            frame_x=args['frame_input'],
                            test_mode=TEST_MODE,
                            train=True)

    net = EMGClassifier(module=model,
                        model_name=name,
                        sub_folder=sub_folder,
                        hyperparamters=args,
                        optimizer=torch.optim.Adam,
                        max_epochs=args['epoch'],
                        lr=args['lr'],
                        dataset=train_set,
                        callbacks=[tensorboard_cb])

    net.fit_with_dataset()

    test_set = CapgDataset(gesture=args['gesture_num'],
                           sequence_len=10,
                           test_mode=TEST_MODE,
                           train=False)

    avg_score = net.test_model(test_set)
    print('test accuracy: {:.4f}'.format(avg_score))


hyperparameters = {
    'input_size': (128,),
    'hidden_size': 256,
    'seq_length': 10,
    'seq_result': False,
    'frame_input': False
}


if __name__ == "__main__":
    test_args = {
        'model': 'lstm',
        'gesture_num': 8,
        'lr': 0.001,
        'lr_step': 5,
        'epoch': 1,
        'train_batch_size': 256,
        'valid_batch_size': 1024,
        'stop_patience': 7,
        'log_interval': 100
    }

    for i in [1, 1.5, 2, 3, 5]:
        hyperparameters['seq_length'] = int(10*i)
        main(test_args, TEST_MODE=False)
