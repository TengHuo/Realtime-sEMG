# -*- coding: UTF-8 -*-
# torch_model.py
# @Time     : 16/May/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
#

import os
import h5py
import torch


import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from emg.utils import CapgDataset
from emg.utils import summary, save_report, save_history_figures


def _get_data_loaders(model_args):
    train_loader = DataLoader(CapgDataset(gesture=model_args['gesture_num'],
                                          sequence_len=model_args['seq_length'],
                                          sequence_result=model_args['seq_result'],
                                          frame_x=model_args['frame_input'],
                                          train=True),
                              batch_size=model_args['train_batch_size'],
                              shuffle=True)

    val_loader = DataLoader(CapgDataset(gesture=model_args['gesture_num'],
                                        sequence_len=model_args['seq_length'],
                                        sequence_result=model_args['seq_result'],
                                        frame_x=model_args['frame_input'],
                                        train=False),
                            batch_size=model_args['val_batch_size'],
                            shuffle=False)

    return train_loader, val_loader


def _calculate_accuracy(_, y, y_pred: torch.Tensor, loss):
    y_pred = F.log_softmax(y_pred, dim=1)
    _, y_pred = torch.max(y_pred, dim=1)
    correct = (y_pred == y).sum().item()
    accuracy = correct / y.size(0)
    return loss.item(), accuracy


def _prepare_folder(model_name, gesture_num):
    # create a folder for storing the model
    root_path = os.path.join(os.sep, *os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-2])
    model_folder = os.path.join(root_path, 'outputs', model_name, '{}'.format(gesture_num))
    # create a folder for this model
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    return model_folder


def start_train(args, model, optimizer, trainer_factory=None,
                evaluator_factory=None, data_loader=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create a folder for storing the model
    args['model_folder'] = _prepare_folder(args['model'], args['gesture_num'])
    params_path = os.path.join(args['model_folder'], 'params.pkl')
    if args['load_model'] and os.path.exists(params_path):
        print('load a pretrained model: {}'.format(args['model']))
        model.load_state_dict(torch.load(params_path))
    else:
        print('train a new model')
    model_summary = summary(model=model,
                            input_size=(args['seq_length'], args['input_size']),
                            batch_size=args['train_batch_size'],
                            device=device)
    print(model_summary)

    # data loader
    if data_loader is None:
        train_loader, val_loader = _get_data_loaders(args)
    else:
        train_loader, val_loader = data_loader(args)

    # create engines
    if trainer_factory is None and evaluator_factory is None:
        trainer_factory = create_supervised_trainer
        evaluator_factory = create_supervised_evaluator
    trainer = trainer_factory(model, optimizer, F.cross_entropy, device=device,
                              output_transform=_calculate_accuracy)
    evaluator = evaluator_factory(model,
                                  metrics={'accuracy': Accuracy(),
                                           'loss': Loss(F.cross_entropy)},
                                  device=device)

    # add progress bar
    pbar = ProgressBar()
    pbar.attach(trainer, output_transform=lambda output: {'loss': round(output[0], 2),
                                                          'accuracy': round(output[1], 2)})

    # configure handlers
    report_log = []
    loss_log = []
    acc_log = []
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer_engine):
        loss_acc = trainer_engine.state.output
        loss_log.append(loss_acc[0])
        acc_log.append(loss_acc[1])

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer_engine):
        loss_acc = trainer_engine.state.output
        # TODO: 改为平均算法
        log = "Epoch {}, Training Results - Avg accuracy: {:.2f} Avg loss: {:.2f}"\
            .format(trainer_engine.state.epoch, loss_acc[1], loss_acc[0])
        print(log)
        report_log.append(log)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer_engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        log = "Epoch {}, Validation Results - Avg accuracy: {:.2f} Avg loss: {:.2f}"\
            .format(trainer_engine.state.epoch, metrics['accuracy'], metrics['loss'])
        print(log)
        report_log.append(log)

    # ----------------------------------------------------------------
    # 修改summary，将summary转为字符串return回来
    # log记录train的loss和accuracy，以及每次val的loss和accuracy（参考keras的history）
    # 3. 生成history的image
    # TODO: 4. 训练完成后test模型
    # 5. store report
    # ----------------------------------------------------------------

    @trainer.on(Events.COMPLETED)
    def save_model(_):
        print('training completed')
        # store model parameters
        torch.save(model.state_dict(), params_path)
        # store report and figures
        report_content = {
            'model_name': args['model'].upper(),
            'hyperparameter': args,
            'model_summary': model_summary,
            'log': report_log,
            'evaluation': 'evaluation result'
        }
        report_path = os.path.join(args['model_folder'], 'report.md')
        save_report(report_content, report_path)
        history_fig_path = os.path.join(args['model_folder'], 'history.png')
        save_history_figures(loss_log, acc_log, history_fig_path)
        # store train history
        train_history_path = os.path.join(args['model_folder'], 'history.h5')
        f = h5py.File(train_history_path, 'w')
        f.create_dataset('loss_history', data=loss_log)
        f.close()

    # add a learning rate scheduler
    step_scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    scheduler = LRScheduler(step_scheduler, save_history=True)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)

    # add early stop handler, if val loss has not improved for patience validation epochs,
    # training will stop
    def score_function(engine):
        val_loss = engine.state.metrics['loss']
        return -val_loss
    handler = EarlyStopping(patience=args['stop_patience'], score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, handler)

    print('start train the model: {}'.format(args['model']))
    trainer.run(train_loader, max_epochs=args['epoch'])

