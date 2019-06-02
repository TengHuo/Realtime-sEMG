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
from torch.utils.tensorboard import SummaryWriter

from emg.models.base import EMGClassifier
from emg.utils import TensorboardCallback, generate_folder
from emg.data_loader.capg_data import CapgDataset


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
            num_layers=3,
            batch_first=True,
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
        return x


def main(train_args):
    # 1. 设置好optimizer
    # 2. 定义好model
    args = {**train_args, **hyperparameters}

    train_set = CapgDataset(gesture=args['gesture_num'],
                            sequence_len=10,
                            sequence_result=False,
                            frame_x=args['frame_input'],
                            TEST=args['test'],
                            train=True)

    x = train_set.data
    y = train_set.targets

    model = LSTM(args['input_size'], args['hidden_size'], args['gesture_num'])
    comment = 'test4'
    f_name = args['model'] + '-' + str(args['gesture_num']) + '-{}'.format(comment)

    # tb_dir = generate_folder(root_folder='tensorboard', folder_name=f_name, sub_folder=comment)
    # writer = SummaryWriter(tb_dir)
    # dummpy_input = torch.ones((1, 10, 128), dtype=torch.float, requires_grad=True)
    # writer.add_graph(model, input_to_model=dummpy_input)

    # tensorboard_cb = TensorboardCallback(writer)

    from skorch import NeuralNetClassifier

    net = NeuralNetClassifier(module=LSTM,
                              criterion=nn.CrossEntropyLoss,
                              optimizer=torch.optim.Adam,
                              lr=args['lr'],
                              max_epochs=args['epoch'],
                              device='cuda',
                              # model_name=f_name,
                              module__input_size=args['input_size'],
                              module__hidden_size=args['hidden_size'],
                              module__output_size=args['gesture_num'])
                              # hyperparamters=args,
                              # lr=args['lr'],
                              # batch_size=args['train_batch_size'],
                              # continue_train=False,
                              # stop_patience=args['stop_patience'],
                              # max_epochs=args['epoch'],
                              # optimizer=torch.optim.Adam)
                              # callbacks=[tensorboard_cb])

    net.fit(x, y)


if __name__ == "__main__":
    test_args = {
        'model': 'lstm',
        'gesture_num': 8,
        'lr': 0.0001,
        'lr_step': 5,
        'epoch': 30,
        'train_batch_size': 256,
        'stop_patience': 7,
        'log_interval': 100,
        'test': False
    }

    main(test_args)
    # print(torch.cuda.is_available())
    # print(torch.cuda.device_count())
    # print(torch.cuda.get_device_name(0))
