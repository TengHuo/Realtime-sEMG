# -*- coding: UTF-8 -*-


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

from emg.utils.tools import init_parameters, generate_folder
from emg.utils.report_logger import ReportLog, save_evaluation
from emg.utils.lr_scheduler import DecayLR
from emg.utils.progressbar import ProgressBar


class EMGSiamese(NeuralNet):
    def __init__(self, module: nn.Module, model_name: str, sub_folder: str,
                 hyperparamters: dict, *args, stop_patience=5,
                 continue_train=False, criterion=nn.CrossEntropyLoss,
                 train_split=CVSplit(cv=0.1), **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(EMGSiamese, self).__init__(module, *args,
                                         device=device,
                                         criterion=criterion,
                                         iterator_train__shuffle=True,
                                         iterator_train__num_workers=4,
                                         iterator_train__batch_size=hyperparamters['train_batch_size'],
                                         iterator_valid__shuffle=False,
                                         iterator_valid__num_workers=4,
                                         iterator_valid__batch_size=hyperparamters['valid_batch_size'],
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
                            patience=self.patience,
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
        return super(EMGSiamese, self).fit(x, y)

    def train_step_single(self, Xi, yi, **fit_params):
        """Compute y_pred, loss value, and update net's gradients.

        The module is set to be in train mode (e.g. dropout is
        applied).

        Parameters
        ----------
        Xi : input data
          A batch of the input data.

        yi : target data
          A batch of the target data.

        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.

        """
        self.module_.train()
        self.optimizer_.zero_grad()
        y_pred = self.infer(Xi, **fit_params)
        loss = self.get_loss(y_pred, yi, X=Xi, training=True)
        loss.backward()

        self.notify(
            'on_grad_computed',
            named_parameters=TeeGenerator(self.module_.named_parameters()),
            X=Xi,
            y=yi
        )

        return {
            'loss': loss,
            'y_pred': y_pred,
            }

    def train_step(self, Xi, yi, **fit_params):
        """Override the function for siamese training,

        Parameters
        ----------
        :param Xi: input data
          A batch of the input data.
        :param yi: target data
          A batch of the target data.
        :param fit_params: dict
          Additional parameters passed to the ``forward`` method of
          the module and to the train_split call.
        :return:
        """
        step_accumulator = self.get_train_step_accumulator()

        def step_fn():
            step = self.train_step_single(Xi, yi, **fit_params)
            step_accumulator.store_step(step)
            return step['loss']

        self.optimizer_.step(step_fn)
        return step_accumulator.get_step()

    def evaluation_step(self, Xi, training=False):
        """Perform a forward step to produce the output used for
        prediction and scoring.

        Therefore the module is set to evaluation mode by default
        beforehand which can be overridden to re-enable features
        like dropout by setting ``training=True``.

        """
        with torch.set_grad_enabled(training):
            self.module_.train(training)
            return self.infer(Xi)

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

    def test_model(self, test_set):
        test_dataloader = DataLoader(dataset=test_set,
                                     batch_size=1024,
                                     shuffle=False,
                                     num_workers=4)
        all_score = []
        for step, (batch_x, batch_y) in enumerate(test_dataloader):
            y_pred = self.evaluation_step(batch_x)
            y_pred = to_numpy(y_pred.max(-1)[-1])
            score = accuracy_score(batch_y, y_pred)
            all_score.append(score)
        avg_score = np.average(all_score)
        save_evaluation(self.model_path, avg_score)
        return avg_score
