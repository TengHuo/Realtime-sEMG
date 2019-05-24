# -*- coding: UTF-8 -*-
# mlp.py
# @Time     : 24/May/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
#


import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

from emg.utils import CapgDataset
from emg.models.torch_model import prepare_folder
from emg.models.torch_model import add_handles


# TODO: 在这里定义模型超参数，运行时合并到输入的参数中


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_data_loaders(gesture_num, train_batch_size, val_batch_size, sequence_len):
    train_loader = DataLoader(CapgDataset(gestures=gesture_num,
                                          sequence_len=sequence_len,
                                          train=True),
                              batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(CapgDataset(gestures=gesture_num,
                                        sequence_len=sequence_len,
                                        train=False),
                            batch_size=val_batch_size, shuffle=False)

    return train_loader, val_loader


def _calculate_accuracy(_, y, y_pred: torch.Tensor, loss):
    y_pred = F.log_softmax(y_pred, dim=1)
    _, y_pred = torch.max(y_pred, dim=1)
    correct = (y_pred == y).sum().item()
    accuracy = correct / y.size(0)
    return loss.item(), accuracy


def run(option, input_size=128, hidden_size=256, seq_length=1):
    # TODO: 这部分代码里只需要完成三件事
    # 1. 加载模型需要的数据
    # 2. 设置好optimizer
    # 3. create trainer和evaluator
    train_loader, val_loader = get_data_loaders(option['gesture_num'],
                                                option['train_batch_size'],
                                                option['val_batch_size'],
                                                seq_length)

    # create a folder for storing the model
    option['model_folder'], option['model_path'] = prepare_folder(option['model'], option['gesture_num'])
    if option['load_model'] and os.path.exists(option['model_path']):
        print('load a pretrained model: {}'.format(option['model']))
        model = torch.load(option['model_path'])
    else:
        print('train a new model')
        model = MLP(input_size, hidden_size, option['gesture_num'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=option['lr'])
    trainer = create_supervised_trainer(model, optimizer, F.cross_entropy, device=device,
                                        output_transform=_calculate_accuracy)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy(),
                                                     'loss': Loss(F.cross_entropy)},
                                            device=device)

    add_handles(model, option, trainer, evaluator, train_loader, val_loader, optimizer)


if __name__ == "__main__":
    args = {
        'model': 'mlp',
        'gesture_num': 8,
        'lr': 0.01,
        'epoch': 10,
        'train_batch_size': 256,
        'val_batch_size': 1024,
        'stop_patience': 5
    }

    run(args)
