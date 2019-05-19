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

from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.handlers import EarlyStopping

from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from torch.optim.lr_scheduler import StepLR

from emg.utils import summary, store_report


def prepare_folder(model_name, gesture_num):
    # create a folder for storing the model
    root_path = os.path.join(os.sep, *os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-2])
    model_folder = os.path.join(root_path, 'outputs', model_name, '{}'.format(gesture_num))
    model_path = os.path.join(model_folder, 'model.pkl')
    # create a folder for this model
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)
    return model_folder, model_path


def add_handles(model, option, trainer, evaluator, train_loader, val_loader, optimizer):
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    pbar = ProgressBar()
    pbar.attach(trainer, ['loss'])

    # ----------------------------------------------------------------
    # TODO: 1. 修改summary，将summary转为字符串return回来
    # TODO: 2. log记录train的loss和accuracy，以及每次val的loss和accuracy（参考keras的history）
    # TODO: 3. 生成history的image
    # TODO: 4. 训练完成后test模型
    # TODO: 5. store report
    model_summary = summary(model, input_size=(10, 128), batch_size=256)
    report_content = {
        'model_name': 'test',
        'hyperparameter': {'test': 1, 'test2': 2},
        'model_summary': model_summary,
        'log': "Training Results - Avg accuracy: 0.10 Avg loss: 0.10\n",
        'history_img_path': './image.png',
        'evaluation': 'evaluation result'
    }
    report_path = ''
    store_report(report_content, report_path)
    # ----------------------------------------------------------------

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer_):
        loss_history.append(trainer_.state.output)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(_):
        print('evaluating the model.....')
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print("Training Results - Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(metrics['accuracy'], metrics['loss']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(_):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print("Validation Results - Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(metrics['accuracy'], metrics['loss']))

    @trainer.on(Events.COMPLETED)
    def save_model(_):
        # BUG: 如果模型被earlystop terminal，则不会保存模型
        print('train completed')
        f = h5py.File(os.path.join(option['model_folder'], 'history.h5'), 'w')
        f.create_dataset('loss_history', data=loss_history)
        f.close()
        torch.save(model, os.path.join(option['model_folder'], 'model.pkl'))

    step_scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    scheduler = LRScheduler(step_scheduler, save_history=True)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)

    def score_function(engine):
        val_loss = engine.state.metrics['loss']
        # print('loss: {}'.format(val_loss))
        return -val_loss

    handler = EarlyStopping(patience=option['stop_patience'], score_function=score_function, trainer=trainer)
    # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
    evaluator.add_event_handler(Events.COMPLETED, handler)

    print('start train the model: {}'.format(option['model']))
    trainer.run(train_loader, max_epochs=option['epoch'])
