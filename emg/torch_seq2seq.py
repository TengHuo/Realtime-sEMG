# -*- coding: UTF-8 -*-


import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from ignite.engine import Events, Engine
from ignite.utils import convert_tensor
from ignite.metrics import Accuracy, Loss

from tqdm import tqdm

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
    train_loader = DataLoader(CapgDataset(gestures=gesture_num,
                                          sequence_len=sequence_len,
                                          sequence_result=True,
                                          train=True),
                              batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(CapgDataset(gestures=gesture_num,
                                        sequence_len=sequence_len,
                                        sequence_result=True,
                                        train=False),
                            batch_size=val_batch_size, shuffle=False)

    return train_loader, val_loader


def _prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options.

    """
    x, y = batch
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking))


def create_trainer(encoder, decoder, encoder_optimizer, decoder_optimizer, loss_fn,
                   device=None, prepare_batch=_prepare_batch):

    if device:
        encoder.to(device)
        decoder.to(device)

    def _update(engine, batch):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        x, y = prepare_batch(batch, device=device)

        encoder_output, hidden = encoder(x)
        decoder_input = torch.zeros((x.size(0), 1, hidden.size(2)), device=device)
        loss = 0.0
        for i in range(encoder_output.size(1)):
            decoder_input, pred, hidden = decoder(decoder_input, hidden)
            loss += loss_fn(pred, y[:, i])

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        return loss.item()

    return Engine(_update)


def create_evaluator(encoder, decoder, metrics,
                     device=None, prepare_batch=_prepare_batch):

    if device:
        encoder.to(device)
        decoder.to(device)

    def _inference(engine, batch):
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            x, y = prepare_batch(batch, device=device)
            encoder_output, hidden = encoder(x)
            decoder_input = torch.zeros((x.size(0), 1, hidden.size(2)), device=device)
            for i in range(encoder_output.size(1)):
                decoder_input, pred, hidden = decoder(decoder_input, hidden)
            return pred, y[:, 0]

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def run(train_batch_size, val_batch_size, input_size, hidden_size, gesture_num,
        seq_length, epochs, lr, log_interval):

    train_loader, val_loader = get_data_loaders(gesture_num,
                                                train_batch_size,
                                                val_batch_size,
                                                seq_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'seq2seq'
    out_size = 8
    root_path = os.path.join(os.sep, *os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-1])
    model_folder = os.path.join(root_path, 'models', model_name, '{}'.format(out_size))
    # create a folder for this model
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)

    encoder_path = os.path.join(model_folder, 'encoder.pkl')
    decoder_path = os.path.join(model_folder, 'decoder.pkl')
    if os.path.exists(encoder_path):
        print('model exist! load it!')
        encoder = torch.load(encoder_path)
    else:
        encoder = EncoderRNN(input_size, hidden_size)

    if os.path.exists(decoder_path):
        print('model exist! load it!')
        decoder = torch.load(decoder_path)
    else:
        decoder = DecoderRNN(hidden_size, gesture_num)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)

    trainer = create_trainer(encoder, decoder, encoder_optimizer, decoder_optimizer,
                             F.cross_entropy, device=device)
    evaluator = create_evaluator(encoder, decoder, device=device,
                                 metrics={'accuracy': Accuracy(),
                                          'cs': Loss(F.cross_entropy)})

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(log_interval)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_cs = metrics['cs']
        tqdm.write(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_cs)
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_cs = metrics['cs']
        tqdm.write(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_cs))

        pbar.n = pbar.last_print_n = 0

    @trainer.on(Events.COMPLETED)
    def save_model(engine):
        print('train completed')
        torch.save(encoder, encoder_path)
        torch.save(decoder, decoder_path)

    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()


if __name__ == "__main__":
    gesture_num = 8
    epoch = 10
    learning_rate = 0.01
    seq_length = 10
    input_size = 128
    hidden_size = 256
    train_batch_size = 256
    val_batch_size = 1024

    run(train_batch_size, val_batch_size, input_size, hidden_size, gesture_num,
        seq_length, epoch, learning_rate, log_interval=10)
