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
from emg.models.torch_model import start_train


hyperparameters = {
    'input_size': (128,),
    'hidden_size': 256,
    'seq_length': 1,
    'seq_result': False,
    'frame_input': False
}


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_size[0], hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main(train_args):
    # 1. 设置好optimizer
    # 2. 定义好model
    args = {**train_args, **hyperparameters}
    model = MLP(args['input_size'], args['hidden_size'], args['gesture_num'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])

    start_train(args, model, optimizer)


if __name__ == "__main__":
    test_args = {
        'model': 'mlp',
        'gesture_num': 8,
        'lr': 0.01,
        'epoch': 10,
        'train_batch_size': 256,
        'val_batch_size': 1024,
        'stop_patience': 5,
        'load_model': False
    }

    main(test_args)
