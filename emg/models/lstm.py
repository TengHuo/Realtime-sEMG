# -*- coding: UTF-8 -*-
# lstm.py
# @Time     : 15/May/2019
# @Auther   : TENG HUO
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


class CapgLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CapgLSTM, self).__init__()

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.rnn(x, None)   # None represents zero initial hidden state
        # choose r_out at the last time step
        x = self.fc(x[:, -1, :])
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


def run(option, input_size=128, hidden_size=256, seq_length=10):
    train_loader, val_loader = get_data_loaders(option['gesture_num'],
                                                option['train_batch_size'],
                                                option['val_batch_size'],
                                                seq_length)

    # create a folder for storing the model
    option['model_folder'], option['model_path'] = prepare_folder(option['model'], option['gesture_num'])
    if os.path.exists(option['model_path']):
        print('seq2seq model exist! load it!')
        model = torch.load(option['model_path'])
    else:
        model = CapgLSTM(input_size, hidden_size, option['gesture_num'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=option['lr'])
    trainer = create_supervised_trainer(model, optimizer, F.cross_entropy, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy(),
                                                     'loss': Loss(F.cross_entropy)},
                                            device=device)

    add_handles(model, option, trainer, evaluator, train_loader, val_loader, optimizer)


if __name__ == "__main__":
    args = {
        'model': 'lstm',
        'gesture_num': 8,
        'lr': 0.01,
        'epoch': 2,
        'train_batch_size': 256,
        'val_batch_size': 1024,
        'stop_patience': 5
    }

    run(args)
