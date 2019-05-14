# -*- coding: UTF-8 -*-

# TODO: 按照MNIST的loader修改capg loader使其符合规范，data可能应该是tensor而不是numpy
# TODO: 训练完成后保存模型，载入模型，记录训练log，记录accuracy
# TODO: 整合代码到main中，修改为按命令行训练

from emg.utils import CapgDataset

from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_

from ignite.engine import Events, Engine, create_supervised_evaluator
from ignite.utils import convert_tensor
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.metrics import Accuracy, Loss

from tqdm import tqdm


class CapgLSTM(nn.Module):
    def __init__(self):
        super(CapgLSTM, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=128,
            hidden_size=64,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(64, 8)

    def forward(self, x):
        r_out, _ = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        predict = self.out(r_out[:, -1, :])
        return predict


def get_data_loaders(train_batch_size, val_batch_size):
    train_loader = DataLoader(CapgDataset(gestures=8, sequence_len=20, train=True),
                              batch_size=train_batch_size, shuffle=True)

    val_loader = DataLoader(CapgDataset(gestures=8, sequence_len=20, train=False),
                            batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader


def _prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options.

    """
    x, y = batch
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking))


def lstm_trainer(model, optimizer, loss_fn,
                 device=None, non_blocking=False,
                 prepare_batch=_prepare_batch):

    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        return loss.item()

    return Engine(_update)


def run(train_batch_size, val_batch_size, epochs, lr, momentum, log_interval):
    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    model = CapgLSTM()
    # model = CapgMLP()
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda'

    optimizer = Adam(model.parameters(), lr=lr)
    # optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    trainer = lstm_trainer(model, optimizer, F.cross_entropy, device=device)

    # step_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    # scheduler = LRScheduler(step_scheduler)
    # trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy(),
                                                     'cs': Loss(F.cross_entropy)},
                                            device=device)

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

    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()


if __name__ == "__main__":
    batch_size = 128
    val_batch_size = 1000
    epochs = 10
    lr = 0.01
    momentum = 0.5
    log_interval = 10
    run(batch_size, val_batch_size, epochs, lr, momentum, log_interval)
