# -*- coding: UTF-8 -*-
# base.py
# @Time     : 31/May/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
#


import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score

from skorch import NeuralNet
from skorch.callbacks import EpochTimer, EpochScoring, BatchScoring
from skorch.callbacks import Checkpoint, TrainEndCheckpoint, EarlyStopping
from skorch.dataset import CVSplit
from skorch.utils import is_dataset, noop, to_numpy
from skorch.utils import train_loss_score, valid_loss_score

from emg.utils import init_parameters, generate_folder, ReportLog, ProgressBar
from emg.utils.report_logger import save_evaluation


class EMGClassifier(NeuralNet):
    def __init__(self, module: nn.Module, model_name: str, sub_folder: str,
                 hyperparamters: dict, *args, stop_patience=5,
                 continue_train=False, criterion=nn.CrossEntropyLoss,
                 train_split=CVSplit(cv=0.1), **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(EMGClassifier, self).__init__(module, *args,
                                            device=device,
                                            criterion=criterion,
                                            train_split=train_split,
                                            **kwargs)
        self.model_name = model_name
        self.hyperparamters = hyperparamters
        self.patience = stop_patience
        self.model_path = generate_folder('checkpoints', model_name, sub_folder=sub_folder)

        if continue_train:
            params = self.model_path + 'train_end_params.pt'
            optimizer = self.model_path + 'train_end_optimizer.pt'
            history = self.model_path + 'train_end_history.json'
            if os.path.exists(params) and os.path.exists(optimizer) and os.path.exists(history):
                print('load parameter from a pretrained model')
                self.load_params(f_params=params,
                                 f_optimizer=optimizer,
                                 f_history=history)
            else:
                raise FileNotFoundError()
        else:
            print('build a new model, init parameters of {}'.format(model_name))
            self.module.apply(init_parameters)

    @property
    def _default_callbacks(self):
        return [
            ('epoch_timer', EpochTimer()),
            ('train_loss', BatchScoring(
                train_loss_score,
                name='train_loss',
                on_train=True,
                target_extractor=noop)),
            ('valid_loss', BatchScoring(
                valid_loss_score,
                name='valid_loss',
                target_extractor=noop)),
            ('valid_acc', EpochScoring(
                'accuracy',
                name='valid_acc',
                lower_is_better=False,)),
            ('checkpoint', Checkpoint(
                dirname=self.model_path)),
            ('end_checkpoint', TrainEndCheckpoint(
                dirname=self.model_path)),
            # ('earlystop',  EarlyStopping(
            #     patience=self.patience,
            #     threshold=1e-4)),
            ('report', ReportLog()),
            ('progressbar', ProgressBar())
        ]

    # pylint: disable=signature-differs
    def check_data(self, X, y):
        if (
                (y is None) and
                (not is_dataset(X)) and
                (self.iterator_train is DataLoader)
        ):
            msg = ("No y-values are given (y=None). You must either supply a "
                   "Dataset as X or implement your own DataLoader for "
                   "training (and your validation) and supply it using the "
                   "``iterator_train`` and ``iterator_valid`` parameters "
                   "respectively.")
            raise ValueError(msg)

    # pylint: disable=arguments-differ
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        if isinstance(self.criterion_, nn.NLLLoss):
            y_pred = torch.log(y_pred)
        return super().get_loss(y_pred, y_true, *args, **kwargs)

    # pylint: disable=signature-differs
    def fit(self, X, y, **fit_params):
        """See ``NeuralNet.fit``.

        In contrast to ``NeuralNet.fit``, ``y`` is non-optional to
        avoid mistakenly forgetting about ``y``. However, ``y`` can be
        set to ``None`` in case it is derived dynamically from
        ``X``.

        """
        # pylint: disable=useless-super-delegation
        # this is actually a pylint bug:
        # https://github.com/PyCQA/pylint/issues/1085
        return super(EMGClassifier, self).fit(X, y, **fit_params)

    def predict_proba(self, X):
        """Where applicable, return probability estimates for
        samples.

        If the module's forward method returns multiple outputs as a
        tuple, it is assumed that the first output contains the
        relevant information and the other values are ignored. If all
        values are relevant, consider using
        :func:`~skorch.NeuralNet.forward` instead.

        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:

            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset

          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.

        Returns
        -------
        y_proba : numpy ndarray

        """
        # Only the docstring changed from parent.
        # pylint: disable=useless-super-delegation
        return super().predict_proba(X)

    def predict(self, X):
        """Where applicable, return class labels for samples in X.

        If the module's forward method returns multiple outputs as a
        tuple, it is assumed that the first output contains the
        relevant information and the other values are ignored. If all
        values are relevant, consider using
        :func:`~skorch.NeuralNet.forward` instead.

        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:

            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset

          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.

        Returns
        -------
        y_pred : numpy ndarray

        """
        y_preds = []
        for yp in self.forward_iter(X, training=False):
            yp = yp[0] if isinstance(yp, tuple) else yp
            y_preds.append(to_numpy(yp.max(-1)[-1]))
        y_pred = np.concatenate(y_preds, 0)
        return y_pred

    def test_model(self, x_test, y_test):
        y_pred = self.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        save_evaluation(self.model_path, score)
        return score
