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
    model = MLP(args['input_size'], args['hidden_size'], args['gesture_num'])
    name = args['name']
    sub_folder = args['sub_folder']

    tensorboard_cb = config_tensorboard(name, sub_folder, model, (1, 128))

    from emg.utils.lr_scheduler import DecayLR
    lr_callback = DecayLR(start_lr=args['lr'], gamma=0.5, step_size=args['lr_step'])

    train_set = CapgDataset(gesture=args['gesture_num'],
                            sequence_len=1,
                            test_mode=TEST_MODE,
                            train=True)

    net = EMGClassifier(module=model,
                        model_name=name,
                        sub_folder=sub_folder,
                        hyperparamters=args,
                        optimizer=torch.optim.Adam,
                        dataset=train_set,
                        callbacks=[tensorboard_cb, lr_callback])

    net.fit_with_dataset()

    test_set = CapgDataset(gesture=args['gesture_num'],
                           sequence_len=1,
                           test_mode=TEST_MODE,
                           train=False)

    avg_score = net.test_model(test_set)
    print('test accuracy: {:.4f}'.format(avg_score))


if __name__ == "__main__":
    test_args = {
        'model': 'mlp',
        'suffix': 'test-shuffle',
        'sub_folder': 'test1',
        'gesture_num': 8,
        'epoch': 300,
        'train_batch_size': 512,
        'valid_batch_size': 2048,
        'lr': 0.001,
        'lr_step': 50}

    print('test')
    default_name = test_args['model'] + '-{}'.format(test_args['suffix'])
    test_args['name'] = default_name
    main(test_args, TEST_MODE=False)
