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
from random import shuffle
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, confusion_matrix

from skorch import NeuralNet
from skorch.callbacks import EpochTimer, EpochScoring, BatchScoring
from skorch.callbacks import Checkpoint, TrainEndCheckpoint, EarlyStopping
from skorch.dataset import CVSplit
from skorch.utils import is_dataset, noop, to_numpy
from skorch.utils import train_loss_score, valid_loss_score
from skorch.dataset import get_len

from emg.utils.tools import init_parameters, generate_folder
from emg.utils.report_logger import ReportLog, save_evaluation
from emg.utils.lr_scheduler import DecayLR
from emg.utils.progressbar import ProgressBar
from emg.data_loader.capg_data import CapgDataset


class EMGClassifier(NeuralNet):
    def __init__(self, module: nn.Module,
                 model_name: str,
                 sub_folder: str,
                 hyperparamters: dict,
                 optimizer,
                 gesture_list: list,  # all gestures index
                 callbacks: list,
                 # train_new_model=True,
                 train_split=CVSplit(cv=0.1, random_state=0)):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(EMGClassifier, self).__init__(module,
                                            criterion=nn.CrossEntropyLoss,
                                            optimizer=optimizer,
                                            lr=hyperparamters['lr'],
                                            max_epochs=hyperparamters['epoch'],
                                            train_split=train_split,
                                            callbacks=callbacks,
                                            device=device,
                                            iterator_train__shuffle=True,
                                            iterator_train__num_workers=4,
                                            iterator_train__batch_size=hyperparamters['train_batch_size'],
                                            iterator_valid__shuffle=False,
                                            iterator_valid__num_workers=4,
                                            iterator_valid__batch_size=hyperparamters['valid_batch_size'])
        self.model_name = model_name
        self.hyperparamters = hyperparamters
        # self.extend_scale = dataset.scale
        self._gesture_mapping = None
        self._all_gestures = gesture_list
        self.module.apply(init_parameters)
        self.model_trained = False
        self.model_path = generate_folder('checkpoints', model_name, sub_folder=sub_folder)

    def init_model_param(self):
        pass

    @property
    def _default_callbacks(self):
        default_cb_list = [
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
            ('report', ReportLog()),
            ('progressbar', ProgressBar())
        ]

        if 'stop_patience' in self.hyperparamters.keys() and \
                self.hyperparamters['stop_patience']:
            earlystop_cb = ('earlystop',  EarlyStopping(
                            patience=self.hyperparamters['stop_patience'],
                            threshold=1e-4))
            default_cb_list.append(earlystop_cb)

        if 'lr_step' in self.hyperparamters.keys() and \
                self.hyperparamters['lr_step']:
            lr_callback = ('lr_schedule', DecayLR(
                           self.hyperparamters['lr'],
                           self.hyperparamters['lr_step'],
                           gamma=0.5))
            default_cb_list.append(lr_callback)

        return default_cb_list

    # pylint: disable=signature-differs
    # def check_data(self, X, y):
    #     if (
    #             (y is None) and
    #             (not is_dataset(X)) and
    #             (self.iterator_train is DataLoader)
    #     ):
    #         msg = ("No y-values are given (y=None). You must either supply a "
    #                "Dataset as X or implement your own DataLoader for "
    #                "training (and your validation) and supply it using the "
    #                "``iterator_train`` and ``iterator_valid`` parameters "
    #                "respectively.")
    #         raise ValueError(msg)

    # pylint: disable=arguments-differ
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        if isinstance(self.criterion_, nn.NLLLoss):
            y_pred = torch.log(y_pred)
        return super().get_loss(y_pred, y_true, *args, **kwargs)

    # pylint: disable=signature-differs
    def fit_with_dataset(self):
        """See ``NeuralNet.fit``.

        In contrast to ``NeuralNet.fit``, ``y`` is non-optional to
        avoid mistakenly forgetting about ``y``. However, ``y`` can be
        set to ``None`` in case it is derived dynamically from
        ``X``.

        """
        # pylint: disable=useless-super-delegation
        # this is actually a pylint bug:
        # https://github.com/PyCQA/pylint/issues/1085
        x = None
        y = None
        super(EMGClassifier, self).fit(x, y)
        self.model_trained = True

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

    def test_model(self, test_gestures, test_set):
        if not self.model_trained:
            params = os.path.join(self.model_path, 'train_end_params.pt')
            optimizer = os.path.join(self.model_path, 'train_end_optimizer.pt')
            history = os.path.join(self.model_path, 'train_end_history.json')
            if os.path.isfile(params) and os.path.isfile(optimizer) and os.path.isfile(history):
                print('load parameter from a pretrained model')
                self.initialize()
                self.load_params(f_params=params,
                                 f_optimizer=optimizer,
                                 f_history=history)
                self.model_trained = True
            else:
                raise FileNotFoundError()

        test_dataloader = DataLoader(dataset=test_set,
                                     batch_size=1024,
                                     shuffle=False,
                                     num_workers=4)
        all_score = []
        y_true_all = []
        y_pred_all = []
        for step, (batch_x, batch_y) in enumerate(test_dataloader):
            y_pred = self.evaluation_step(batch_x)
            y_pred = to_numpy(y_pred.max(-1)[-1])
            y_true_all += batch_y.tolist()
            y_pred_all += y_pred.tolist()
            score = accuracy_score(batch_y, y_pred)
            all_score.append(score)
        avg_score = np.average(all_score)
        labels = [self.gesture_map[str(i)] for i in test_gestures]
        matrix = confusion_matrix(y_true_all, y_pred_all, labels)
        print(matrix)
        save_evaluation(self.model_path, str(test_gestures), matrix, avg_score)
        return avg_score

    @property
    def gesture_map(self):
        if self._gesture_mapping is None:
            map_file = os.path.join(self.model_path, 'gesture_map.json')
            if os.path.isfile(map_file):
                with open(map_file, 'r') as f:
                    self._gesture_mapping = json.load(f)
            else:
                self._gesture_mapping = {str(self._all_gestures[i]): i for i in range(len(self._all_gestures))}
                with open(map_file, 'w') as f:
                    json.dump(self._gesture_mapping, f)
        return self._gesture_mapping

    # @gesture_map.setter
    # def gesture_map(self, gesture_list):
    #     self._gesture_mapping = {gesture_list[i]: i for i in range(len(gesture_list))}
    #     map_file = os.path.join(self.model_path, 'gesture_map.json')
    #     with open(map_file, 'wb') as f:
    #         json.dump(self._gesture_mapping, f)

    def fit_loop(self, X, y=None, epochs=None, **fit_params):
        epochs = epochs if epochs is not None else self.max_epochs

        # split K-fold dataset indcies
        dataset = self.get_dataset(X, y)
        k = 10
        fold_indcies = self.split_k_fold(k, len(dataset))

        for e in range(epochs):
            # get train and validation set
            valid_fold_idx = e % k
            idx_train = []
            for i in range(k):
                if i == valid_fold_idx:
                    continue
                else:
                    idx_train += fold_indcies[i]
            idx_train = np.array(idx_train, dtype=int)
            idx_valid = np.array(fold_indcies[valid_fold_idx], dtype=int)

            dataset_train = torch.utils.data.Subset(dataset, idx_train)
            dataset_valid = torch.utils.data.Subset(dataset, idx_valid)
            on_epoch_kwargs = {
                'dataset_train': dataset_train,
                'dataset_valid': dataset_valid,
            }

            self.notify('on_epoch_begin', **on_epoch_kwargs)
            train_batch_count = 0
            for data in self.get_iterator(dataset_train, training=True):
                xi, yi = data
                self.notify('on_batch_begin', X=xi, y=yi, training=True)
                step = self.train_step(xi, yi, **fit_params)
                self.history.record_batch('train_loss', step['loss'].item())
                self.history.record_batch('train_batch_size', get_len(xi))
                self.notify('on_batch_end', X=xi, y=yi, training=True, **step)
                train_batch_count += 1
            self.history.record("train_batch_count", train_batch_count)

            valid_batch_count = 0
            for data in self.get_iterator(dataset_valid, training=False):
                xi, yi = data
                self.notify('on_batch_begin', X=xi, y=yi, training=False)
                step = self.validation_step(xi, yi, **fit_params)
                self.history.record_batch('valid_loss', step['loss'].item())
                self.history.record_batch('valid_batch_size', get_len(xi))
                self.notify('on_batch_end', X=xi, y=yi, training=False, **step)
                valid_batch_count += 1
            self.history.record("valid_batch_count", valid_batch_count)

            self.notify('on_epoch_end', **on_epoch_kwargs)
        return self

    def split_k_fold(self, k, length):
        scale = self.dataset.scale
        true_len = length // scale
        folds = [[] for _ in range(k)]
        data_indices = list(range(true_len))
        shuffle(data_indices)
        for i in range(true_len):
            fold_idx = i % k
            extend_idx = data_indices[i] * scale
            extend_idcies = [extend_idx + j for j in range(scale)]
            folds[fold_idx] += extend_idcies
        return folds
