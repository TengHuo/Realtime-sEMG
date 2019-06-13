# -*- coding: UTF-8 -*-


from skorch import NeuralNet
from skorch.callbacks import Callback


class DecayLR(Callback):
    """Base class for callbacks.

        All custom callbacks should inherit from this class. The subclass
        may override any of the ``on_...`` methods. It is, however, not
        necessary to override all of them, since it's okay if they don't
        have any effect.

        Classes that inherit from this also gain the ``get_params`` and
        ``set_params`` method.

        """

    def __init__(self, start_lr, step_size, gamma):
        self.lr = start_lr
        self.gamma = gamma
        self.step_size = step_size
        self.__batch = 0

    def on_train_begin(self, net,
                       X=None, y=None, **kwargs):
        """Called at the beginning of training."""
        self.__batch = 0

    def on_epoch_begin(self, net,
                       dataset_train=None, dataset_valid=None, **kwargs):
        """Called at the beginning of each epoch."""
        self.__batch += 1

    def on_epoch_end(self, net: NeuralNet,
                     dataset_train=None, dataset_valid=None, **kwargs):
        """Called at the end of each epoch."""
        if self.__batch % self.step_size == 0:
            for g in net.optimizer_.param_groups:
                self.lr *= self.gamma
                g['lr'] = self.lr
