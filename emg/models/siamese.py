# -*- coding: UTF-8 -*-


import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.modules.distance import PairwiseDistance

from sklearn.metrics import accuracy_score

from skorch import NeuralNet
from skorch.callbacks import EpochTimer, EpochScoring, BatchScoring
from skorch.callbacks import Checkpoint, TrainEndCheckpoint, EarlyStopping
from skorch.dataset import CVSplit
from skorch.utils import is_dataset, noop, to_numpy
from skorch.utils import train_loss_score, valid_loss_score
from skorch.utils import TeeGenerator
from skorch.utils import to_tensor

from emg.utils.tools import init_parameters, generate_folder
from emg.utils.report_logger import ReportLog, save_evaluation
from emg.utils.lr_scheduler import DecayLR
from emg.utils.progressbar import ProgressBar

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class SiameseEMG(NeuralNet):
    def __init__(self,
                 module: nn.Module,
                 model_name: str,
                 sub_folder: str,
                 hyperparamters: dict,
                 optimizer,
                 gesture_list: list,
                 callbacks: list,
                 train_split=CVSplit(cv=0.1, random_state=0)):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(SiameseEMG, self).__init__(module,
                                         criterion=nn.TripletMarginLoss,
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
        self.distance = PairwiseDistance().to(self.device)
        self.model_path = generate_folder('checkpoints', model_name, sub_folder=sub_folder)

        self.clf = KNeighborsClassifier(n_neighbors=5)
        self.clf_fit = False
        self.anchors = []
        self.labels = []

        print('build a new model, init parameters of {}'.format(model_name))
        param_dict = self.load_pretrained("pretrained_end_params.pt")
        self.module.load_state_dict(param_dict)

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
            # ('checkpoint', Checkpoint(
            #     dirname=self.model_path)),
            # ('end_checkpoint', TrainEndCheckpoint(
            #     dirname=self.model_path)),
            ('report', ReportLog()),
            ('progressbar', ProgressBar())
        ]

        # if 'stop_patience' in self.hyperparamters.keys() and \
        #         self.hyperparamters['stop_patience']:
        #     earlystop_cb = ('earlystop',  EarlyStopping(
        #                     patience=self.patience,
        #                     threshold=1e-4))
        #     default_cb_list.append(earlystop_cb)
        #
        # if 'lr_step' in self.hyperparamters.keys() and \
        #         self.hyperparamters['lr_step']:
        #     lr_callback = ('lr_schedule', DecayLR(
        #                    self.hyperparamters['lr'],
        #                    self.hyperparamters['lr_step'],
        #                    gamma=0.5))
        #     default_cb_list.append(lr_callback)

        return default_cb_list

    def load_pretrained(self, param_file: str):
        dict_path = os.path.join(self.model_path, "..", param_file)
        pretrained_dict = torch.load(dict_path, map_location=self.device)
        model_dict = self.module.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        print("load pretrained parameters from: {}".format(dict_path))
        return model_dict

    def triplet_infer(self, x):
        """Perform a single inference step on a batch of data.

        Parameters
        ----------
        x : input data
          A batch of the input data.

        """
        x = to_tensor(x, device=self.device)
        return self.module_(x[0], x[1], x[2])

    def get_triplet_loss(self, anchor, postive, negative):
        return self.criterion_(anchor, postive, negative)

    def validation_step(self, Xi, yi, **fit_params):
        """Perform a forward step using batched data and return the
        resulting loss.

        The module is set to be in evaluation mode (e.g. dropout is
        not applied).

        Parameters
        ----------
        Xi : input data
          A batch of the input data.

        yi : Not used in this method

        **fit_params : Not used in this method

        """
        self.module_.eval()
        with torch.no_grad():
            anchor, positive, negative = self.triplet_infer(Xi)
            loss = self.get_triplet_loss(anchor, positive, negative)

            # y_pred = self.euclidean_distance(anchor, positive, negative)
            if not self.clf_fit:
                self.labels = self.labels.ravel()
                self.clf.fit(self.anchors, self.labels)
                self.clf_fit = True
            embedded = positive.data.cpu().numpy()
            y_pred = self.clf.predict(embedded)
            # print(y_pred.shape)
            y_pred = torch.from_numpy(y_pred)

        return {
            'loss': loss,
            'y_pred': y_pred}

    def siamese_train_step_single(self, xi, yi):
        """Compute y_pred, loss value, and update net's gradients.
        siamese training
        The module is set to be in train mode (e.g. dropout is
        applied).

        Parameters
        ----------
        xi : input data
          A batch of the input data.

        yi : input data
          A batch of the input data.

        """
        self.module_.train()
        self.optimizer_.zero_grad()
        anchor, positive, negative = self.triplet_infer(xi)

        # y_pred = self.euclidean_distance(anchor, positive, negative)
        self.anchors = anchor.data.cpu().numpy()
        self.labels = yi.data.cpu().numpy()
        self.clf_fit = False

        loss = self.get_triplet_loss(anchor, positive, negative)
        loss.backward()

        self.notify(
            'on_grad_computed',
            named_parameters=TeeGenerator(self.module_.named_parameters()),
            X=xi,
            y=yi
        )

        return {'loss': loss,
                'y_pred': None}

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
            step = self.siamese_train_step_single(Xi, yi)
            step_accumulator.store_step(step)
            return step['loss']

        self.optimizer_.step(step_fn)
        return step_accumulator.get_step()

    def fit_with_dataset(self):
        x = None
        y = None
        return super(SiameseEMG, self).fit(x, y)
