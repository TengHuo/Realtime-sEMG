# -*- coding: UTF-8 -*-
# torch_seq2seq.py
# @Time     : 15/May/2019
# @Auther   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
# TODO: 3. 模型优化：1. 增加dropout，2. 尝试bn，3. 增加模型复杂度

import os
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import EarlyStopping
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from torch.optim.lr_scheduler import StepLR

from emg.utils import CapgDataset


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


def run(train_batch_size, val_batch_size, input_size, hidden_size, gesture_num,
        seq_length, epochs, lr):

    train_loader, val_loader = get_data_loaders(gesture_num,
                                                train_batch_size,
                                                val_batch_size,
                                                seq_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a folder for storing the model
    model_name = 'seq2seq'
    out_size = 8
    root_path = os.path.join(os.sep, *os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-1])
    model_folder = os.path.join(root_path, 'models', model_name, '{}'.format(out_size))
    # create a folder for this model
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)

    model_path = os.path.join(model_folder, 'transformer.pkl')
    # check model files
    if False:#os.path.exists(model_path):
        print('model exist! load it!')
        model = torch.load(model_path)
    else:
        model = Transformer(input_size, hidden_size, gesture_num)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = create_supervised_trainer(model, optimizer, F.cross_entropy, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy(),
                                                     'loss': Loss(F.cross_entropy)},
                                            device=device)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    pbar = ProgressBar()
    pbar.attach(trainer, ['loss'])

    loss_history = []
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        loss_history.append(trainer.state.output)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        print('evaluating the model.....')
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print("Training Results - Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(metrics['accuracy'], metrics['loss']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print("Validation Results - Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(metrics['accuracy'], metrics['loss']))

    @trainer.on(Events.COMPLETED)
    def save_model(trainer):
        print('train completed')
        f = h5py.File('../models/history.h5', 'w')
        f.create_dataset('loss_history', data=loss_history)
        f.close()
        torch.save(model, model_path)

    step_scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    scheduler = LRScheduler(step_scheduler, save_history=True)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)

    def score_function(engine):
        val_loss = engine.state.metrics['loss']
        # print('loss: {}'.format(val_loss))
        return -val_loss

    handler = EarlyStopping(patience=2, score_function=score_function, trainer=trainer)
    # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
    evaluator.add_event_handler(Events.COMPLETED, handler)

    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()


if __name__ == "__main__":
    epoch = 5
    learning_rate = 0.01
    seq_length = 10
    input_size = 128
    hidden_size = 256
    gesture_num = 8
    train_batch_size = 256
    val_batch_size = 1024

    run(train_batch_size, val_batch_size, input_size, hidden_size, gesture_num,
        seq_length, epoch, learning_rate)
