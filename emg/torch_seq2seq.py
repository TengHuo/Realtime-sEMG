# -*- coding: UTF-8 -*-


from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import torch.nn.functional as F

from ignite.engine import Events, Engine, create_supervised_evaluator
from ignite.utils import convert_tensor
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.metrics import Accuracy, Loss

from emg.utils import CapgDataset


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()

        self.rnn = nn.GRU(         # if use nn.RNN(), it hardly learns
            input_size=input_size,
            hidden_size=hidden_size,         # rnn hidden unit
            num_layers=1,           # number of rrnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

    def forward(self, x):
        output, hidden = self.rnn(x, None)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()

        self.rnn = nn.GRU(         # if use nn.RNN(), it hardly learns
            input_size=hidden_size,
            hidden_size=hidden_size,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, prev_hidden):
        output, _ = self.rnn(x, prev_hidden)
        return self.fc(output[:, -1, :])


def get_data_loaders(gesture_num, train_batch_size, val_batch_size, sequence_len):
    train_loader = DataLoader(CapgDataset(gestures=gesture_num, sequence_len=sequence_len, sequence_result=False,
                                          train=True), batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(CapgDataset(gestures=gesture_num, sequence_len=sequence_len, sequence_result=False,
                                        train=False), batch_size=val_batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    gesture_num = 8
    epoch = 10
    learning_rate = 0.01
    seq_length = 10
    input_size = 128
    hidden_size = 256
    train_batch_size = 64
    val_batch_size = 1000

    encoder = EncoderRNN(input_size, hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, gesture_num).to(device)

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()

    train_loader, _ = get_data_loaders(gesture_num, train_batch_size, val_batch_size, seq_length)

    for e in range(epoch):  # loop over the dataset multiple times

        loss100 = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            x, labels = data

            encoder_output, hidden = encoder(x)
            pred = decoder(encoder_output, hidden)
            # print(pred.size())

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # test1 = pred_seq[:, 0, :]
            # test2 = labels[:, 0]
            # print(pred_seq[:, 0, :].size())
            # print(labels[:, 0].size())
            loss = criterion(pred, labels)

            # loss = 0.0
            # for i in range(pred_seq.size(1)):
            #     loss += criterion(pred_seq[:, i, :], labels[:, i])

            # print(loss)
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            loss100 += loss.item()
            if i % 100 == 99:
                print('[Epoch %d, Batch %5d] loss: %.3f' %
                      (epoch + 1, i + 1, loss100 / 100))
                loss100 = 0.0
print("Done Training!")

