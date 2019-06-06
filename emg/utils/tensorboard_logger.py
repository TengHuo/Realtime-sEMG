# -*- coding: UTF-8 -*-
# tensorboard_logger.py
# @Time     : 31/May/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
#

import torch
from skorch import NeuralNet
from torch.utils.tensorboard import SummaryWriter
from skorch.callbacks.base import Callback
from emg.utils.tools import generate_folder


def config_tensorboard(folder_name, sub_folder, model=None, dummy_input_size=None):
    tb_dir = generate_folder('tensorboard', folder_name, sub_folder)
    writer = SummaryWriter(tb_dir)
    if model and dummy_input_size:
        dummy_input = torch.ones(dummy_input_size, dtype=torch.float,
                                 requires_grad=True)
        writer.add_graph(model, input_to_model=dummy_input)
    return TensorboardCallback(writer)


class TensorboardCallback(Callback):
    """Base class for callbacks.

    All custom callbacks should inherit from this class. The subclass
    may override any of the ``on_...`` methods. It is, however, not
    necessary to override all of them, since it's okay if they don't
    have any effect.

    Classes that inherit from this also gain the ``get_params`` and
    ``set_params`` method.

    """

    def __init__(self, tb_writer: SummaryWriter, scalar_reduction=torch.norm):
        self.writer = tb_writer
        self.reduction = scalar_reduction
        self.batch_step = 0
        self.epoch_step = 0

    def initialize(self):
        """(Re-)Set the initial state of the callback. Use this
        e.g. if the callback tracks some state that should be reset
        when the model is re-initialized.

        This method should return self.

        """
        return self

    def on_train_begin(self, net: NeuralNet,
                       X=None, y=None, **kwargs):
        """Called at the beginning of training."""
        self.batch_step = 0
        self.epoch_step = 0

    def on_train_end(self, net: NeuralNet,
                     X=None, y=None, **kwargs):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, net: NeuralNet,
                       dataset_train=None, dataset_valid=None, **kwargs):
        """Called at the beginning of each epoch."""
        self.epoch_step += 1

    def on_epoch_end(self, net: NeuralNet,
                     dataset_train=None, dataset_valid=None, **kwargs):
        """Called at the end of each epoch."""
        params = net.optimizer_.param_groups
        for i in range(len(params)):
            self.writer.add_scalar('learning_rate/lr_{}'.format(i),
                                   scalar_value=params[i]['lr'],
                                   global_step=self.epoch_step)

        valid_acc = net.history[-1, 'valid_acc']
        self.writer.add_scalar("validation/accuracy",
                               scalar_value=valid_acc,
                               global_step=self.epoch_step)
        valid_loss = net.history[-1, 'valid_loss']
        self.writer.add_scalar("validation/loss",
                               scalar_value=valid_loss,
                               global_step=self.epoch_step)

        for name, p in net.module_.named_parameters():
            name = name.replace('.', '/')
            self.writer.add_histogram(tag="weights/{}".format(name),
                                      values=p.data.detach().cpu().numpy(),
                                      global_step=self.epoch_step)
            self.writer.add_histogram(tag="grads/{}".format(name),
                                      values=p.grad.detach().cpu().numpy(),
                                      global_step=self.epoch_step)

    def on_batch_begin(self, net: NeuralNet,
                       Xi=None, yi=None, training=None, **kwargs):
        """Called at the beginning of each batch."""
        if training:
            self.batch_step += 1

    def on_batch_end(self, net: NeuralNet,
                     Xi=None, yi=None, training=None, **kwargs):
        """Called at the end of each batch."""
        batch_info = net.history[-1, 'batches', -1]
        if training:
            self.writer.add_scalar("training/loss",
                                   scalar_value=batch_info['train_loss'],
                                   global_step=self.batch_step)

    def on_grad_computed(self, net: NeuralNet, named_parameters,
                         Xi=None, yi=None, training=None, **kwargs):
        """Called once per batch after gradients have been computed but before
        an update step was performed.
        """
        for name, p in named_parameters:
            name = name.replace('.', '/')
            self.writer.add_scalar("weights_{}/{}".format(self.reduction.__name__, name),
                                   self.reduction(p.data),
                                   global_step=self.batch_step)
            self.writer.add_scalar("grads_{}/{}".format(self.reduction.__name__, name),
                                   self.reduction(p.grad),
                                   global_step=self.batch_step)
