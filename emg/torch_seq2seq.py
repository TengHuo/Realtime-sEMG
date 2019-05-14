# -*- coding: UTF-8 -*-


import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from emg.utils import CapgDataset


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x):
        output, hidden = self.rnn(x, None)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()

        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, prev_hidden):
        output, hidden = self.rnn(x, prev_hidden)
        return output, self.fc(output[:, 0, :]), hidden


def get_data_loaders(gesture_num, train_batch_size, val_batch_size, sequence_len):
    train_loader = DataLoader(CapgDataset(gestures=gesture_num, sequence_len=sequence_len, sequence_result=True,
                                          train=True), batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(CapgDataset(gestures=gesture_num, sequence_len=sequence_len, sequence_result=True,
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

        avg_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            x, labels = data

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            _, hidden = encoder(x)
            decoder_input = torch.zeros((train_batch_size, 1, hidden_size), device=device)
            loss = 0.0
            for i in range(seq_length):
                decoder_input, pred, hidden = decoder(decoder_input, hidden)
                loss += criterion(pred, labels[:, i])
            # print(loss)

            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            avg_loss += loss.item()
            if i % 100 == 99:
                print('[Epoch %d, Batch %5d] loss: %.3f' %
                      (epoch, i + 1, avg_loss / 100))
                avg_loss = 0.0
print("Done Training!")

