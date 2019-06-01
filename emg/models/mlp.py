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
from torch.utils.tensorboard import SummaryWriter

from emg.models.base import EMGClassifier
from emg.utils import TensorboardCallback, generate_folder
from emg.data_loader.capg_data import CapgDataset


hyperparameters = {
    'input_size': (128,),
    'hidden_size': 256,
    'seq_result': False,
    'frame_input': False
}


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # an affine operation: y = Wx + b
        self.bn1 = nn.BatchNorm1d(input_size[0])
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(input_size[0], hidden_size)
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


def main(train_args):
    # 1. 设置好optimizer
    # 2. 定义好model
    args = {**train_args, **hyperparameters}

    train_set = CapgDataset(gesture=args['gesture_num'],
                            sequence_len=1,
                            sequence_result=False,
                            frame_x=args['frame_input'],
                            TEST=args['test'],
                            train=True)

    x = train_set.data
    y = train_set.targets

    model = MLP(args['input_size'], args['hidden_size'], args['gesture_num'])
    f_name = args['model'] + '-' + str(args['gesture_num']) + '-testearlystop'

    tb_dir = generate_folder(root_folder='tensorboard', folder_name=f_name, sub_folder='3fc-2bn')
    writer = SummaryWriter(tb_dir)
    dummpy_input = torch.ones((1, 128), dtype=torch.float, requires_grad=True)
    writer.add_graph(model, input_to_model=dummpy_input)

    tensorboard_cb = TensorboardCallback(writer)
    net = EMGClassifier(module=model, model_name=f_name,
                        hyperparamters=args,
                        lr=args['lr'],
                        batch_size=args['train_batch_size'],
                        continue_train=False,
                        stop_patience=args['stop_patience'],
                        max_epochs=args['epoch'],
                        optimizer=torch.optim.Adam,
                        callbacks=[tensorboard_cb])

    net.fit(x, y)


if __name__ == "__main__":
    test_args = {
        'model': 'mlp',
        'gesture_num': 8,
        'lr': 0.001,
        'lr_step': 5,
        'epoch': 30,
        'train_batch_size': 1024,
        'val_batch_size': 2048,
        'stop_patience': 5,
        'log_interval': 100,
        'test': True
    }

    main(test_args)
