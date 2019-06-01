
import os
import numpy as np
import torch
import h5py
from datetime import datetime

from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers.tensorboard_logger import *

from emg.utils import summary, save_report, save_history_figures


class Manager(object):
    def __init__(self, args: dict, model: nn.Module, data_loader, parameter_file=None):
        self.__train_history = []
        self.__eval_history = []
        self.__report_log = []
        self.__trainer = None
        self.__evaluator = None
        self.__train_set, self.__val_set = data_loader(args)

        self.timestamp = datetime.now().strftime('%m-%d-%H-%M-%S')
        args['checkpoint_folder'] = _prepare_checkpoints_folder(args['model'], sub_folder=args['gesture_num'])
        if parameter_file is None:
            # init model with xav
            print('train a new model')
            model.apply(Manager.init_parameters)
        else:
            params_path = os.path.join(args['checkpoint_folder'], parameter_file)
            print('load a pretrained model: {}'.format(args['model']))
            model.load_state_dict(torch.load(params_path))

        tb_path = os.path.join(args['checkpoint_folder'], 'log', self.timestamp)
        self.__tb_logger = TensorboardLogger(log_dir=tb_path)

        self.args = args
        self.model = model

        if 'seq_length' in args:
            x_size = (args['seq_length'], *args['input_size'])
        else:
            # if it is not sequence input data, seq_length will be None
            x_size = args['input_size']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_summary = summary(model=self.model, input_size=x_size,
                                     batch_size=self.args['train_batch_size'],
                                     device=self.device)

    @staticmethod
    def init_parameters(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.RNNBase):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

    def compile(self, optimizer, loss=F.cross_entropy,
                trainer_generator=None, evaluator_generator=None):
        # 调用attch方法config trainer和evaluator
        # 如果在fit之前没有compile就抛出异常
        if trainer_generator is None:
            trainer_generator = create_supervised_trainer
        if evaluator_generator is None:
            evaluator_generator = create_supervised_evaluator
        trainer = trainer_generator(self.model, optimizer, loss, device=self.device,
                                    output_transform=_default_train_output)
        evaluator = evaluator_generator(self.model, device=self.device,
                                        metrics={'accuracy': Accuracy(),
                                                 'loss': Loss(loss)})
        self.__attach_logger_(trainer, evaluator)
        self.__attach_handlers_(trainer, evaluator, optimizer)
        self.__attach_tensorboard_(trainer, evaluator, optimizer)

        self.__trainer = trainer
        self.__evaluator = evaluator

    def __attach_logger_(self, trainer, evaluator):
        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda loss_acc: {'loss': round(loss_acc[0], 2),
                                                                'accuracy': round(loss_acc[1], 2)})

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(trainer_engine):
            loss_acc = trainer_engine.state.output
            iter_index = trainer_engine.state.iteration
            if iter_index % self.args['log_interval'] == 1:
                self.__train_history.append([iter_index, loss_acc[0], loss_acc[1]])

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer_engine):
            evaluator.run(self.__val_set)
            metrics = evaluator.state.metrics
            iter_index = trainer_engine.state.iteration
            loss = metrics['loss']
            acc = metrics['accuracy']
            log = "Epoch {}, Validation Results - Avg accuracy: {:.2f} Avg loss: {:.2f}" \
                .format(trainer_engine.state.epoch, acc, loss)
            print(log)
            self.__report_log.append(log)
            self.__eval_history.append([iter_index, loss, acc])

    def __attach_handlers_(self, trainer, evaluator, optimizer):
        handler = ModelCheckpoint(dirname=self.args['checkpoint_folder'], filename_prefix=self.args['model'],
                                  save_interval=1, n_saved=2, save_as_state_dict=True)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {self.args['gesture_num']: self.model})

        # add a learning rate scheduler
        # step_scheduler = StepLR(optimizer, step_size=self.args['lr_step'], gamma=0.1)
        # scheduler = LRScheduler(step_scheduler, save_history=True)
        # trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)

        # add early stop handler, if val loss has not improved for patience validation epochs,
        # training will stop
        def score_function(engine):
            val_loss = engine.state.metrics['loss']
            return -val_loss
        handler = EarlyStopping(patience=self.args['stop_patience'],
                                score_function=score_function,
                                trainer=trainer)
        evaluator.add_event_handler(Events.COMPLETED, handler)

    def __attach_tensorboard_(self, trainer, evaluator, optimizer):
        # Attach the logger to the trainer to log training loss at each iteration
        self.__tb_logger.attach(trainer, event_name=Events.ITERATION_COMPLETED,
                                log_handler=OutputHandler(tag="training",
                                                          output_transform=lambda loss_acc: {'loss': loss_acc[0],
                                                                                             'accuracy': loss_acc[1]}))

        # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
        # each epoch. We setup `another_engine=trainer` to take the epoch of the `trainer` instead of `evaluator`.
        self.__tb_logger.attach(evaluator, event_name=Events.EPOCH_COMPLETED,
                                log_handler=OutputHandler(tag="validation",
                                                          metric_names=["loss", "accuracy"],
                                                          another_engine=trainer))

        # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
        self.__tb_logger.attach(trainer, event_name=Events.EPOCH_COMPLETED,
                                log_handler=OptimizerParamsHandler(optimizer))

        # Attach the logger to the trainer to log model's weights norm after each iteration
        self.__tb_logger.attach(trainer, event_name=Events.ITERATION_COMPLETED,
                                log_handler=WeightsScalarHandler(self.model))

        # Attach the logger to the trainer to log model's weights as a histogram after each epoch
        self.__tb_logger.attach(trainer, event_name=Events.EPOCH_COMPLETED,
                                log_handler=WeightsHistHandler(self.model))

        # # Attach the logger to the trainer to log model's gradients norm after each iteration
        self.__tb_logger.attach(trainer, event_name=Events.ITERATION_COMPLETED,
                                log_handler=GradsScalarHandler(self.model))

        # Attach the logger to the trainer to log model's gradients as a histogram after each epoch
        self.__tb_logger.attach(trainer, event_name=Events.EPOCH_COMPLETED,
                                log_handler=GradsHistHandler(self.model))

    def summary(self):
        print(self.model_summary)

    def start_train(self):
        if self.__trainer is None or self.__evaluator is None:
            raise RuntimeError('model not compile')
        print('start train the model: {}'.format(self.args['model']))
        self.__trainer.run(self.__train_set, max_epochs=self.args['epoch'])
        return self.history

    def test(self, tester, test_set):
        raise NotImplementedError('not implement')

    @property
    def history(self):
        return np.array(self.__train_history), np.array(self.__eval_history)

    def finish(self):
        # generate a report and save figures
        print('train complete')
        report_content = {
            'datetime': self.timestamp,
            'model_name': self.args['model'].upper(),
            'model_summary': self.model_summary,
            'hyperparameter': self.args,
            'log': self.__report_log,
            'evaluation': 'evaluation result'
        }
        report_path = os.path.join(self.args['checkpoint_folder'], 'report.md')
        save_report(report_content, report_path)
        # generate a firgure of loss and accuracy
        history_fig_path = os.path.join(self.args['checkpoint_folder'], 'history.png')
        train_history, eval_history = self.history
        save_history_figures(train_history, eval_history, history_fig_path)
        # store loss history and accuracy history
        history_path = os.path.join(self.args['checkpoint_folder'], 'history.h5')
        f = h5py.File(history_path, 'w')
        f.create_dataset('train_history', data=train_history)
        f.create_dataset('eval_history', data=eval_history)
        f.close()
        self.__tb_logger.close()


def _prepare_checkpoints_folder(model_name, sub_folder):
    # create a folder for storing the model
    root_path = os.path.join(os.sep, *os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-2])
    model_folder = os.path.join(root_path, 'checkpoints', '{}-{}'.format(model_name, sub_folder))
    # create a folder for this model
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    return model_folder


def _default_train_output(_, y, y_pred: torch.Tensor, loss):
    y_pred = F.log_softmax(y_pred, dim=1)
    _, y_pred = torch.max(y_pred, dim=1)
    correct = (y_pred == y).sum().item()
    accuracy = correct / y.size(0)
    return loss.item(), accuracy
