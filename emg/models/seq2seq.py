# -*- coding: UTF-8 -*-
# seq2seq.py
# @Time     : 15/May/2019
# @Auther   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
# TODO: 3. 模型优化：1. 增加dropout，2. 尝试bn，3. 增加模型复杂度

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

from emg.utils import CapgDataset
from .torch_model import add_handles, prepare_folder


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x):
        _, hidden = self.rnn(x, None)
        return hidden


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()

        self.fc = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc(x))
        return self.output(x)


class Transformer(nn.Module):
    def __init__(self, input_length, hidden_size, output_size):
        super(Transformer, self).__init__()

        self.encoder = Encoder(input_length, hidden_size)
        self.decoder = Decoder(hidden_size, output_size)

    def forward(self, x):
        hidden = self.encoder(x)
        return self.decoder(hidden[0])


def get_data_loaders(gesture_num, train_batch_size, val_batch_size, sequence_len):
    train_loader = DataLoader(CapgDataset(gestures=gesture_num,
                                          sequence_len=sequence_len,
                                          sequence_result=False,
                                          train=True),
                              batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(CapgDataset(gestures=gesture_num,
                                        sequence_len=sequence_len,
                                        sequence_result=False,
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
        model = Transformer(input_size, hidden_size, option['gesture_num'])

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
        'model': 'seq2seq',
        'gesture_num': 8,
        'lr': 0.01,
        'epoch': 5,
        'train_batch_size': 256,
        'val_batch_size': 1024,
        'stop_patience': 5
    }

    run(args)
