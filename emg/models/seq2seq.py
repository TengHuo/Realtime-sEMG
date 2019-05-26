# -*- coding: UTF-8 -*-
# seq2seq.py
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


def main(train_args):
    # 1. 设置好optimizer
    # 2. 定义好model
    args = {**train_args, **hyperparameters}
    model = Transformer(args['input_size'], args['hidden_size'], args['gesture_num'])
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'])

    start_train(args, model, optimizer)


if __name__ == "__main__":
    test_args = {
        'model': 'seq2seq',
        'gesture_num': 8,
        'lr': 0.01,
        'epoch': 10,
        'train_batch_size': 256,
        'val_batch_size': 1024,
        'stop_patience': 5,
        'load_model': False
    }

    main(test_args)