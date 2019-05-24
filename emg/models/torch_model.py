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
import matplotlib.pyplot as plt

from ignite.engine import Events
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


def _plot_loss_history(loss_history, acc_history, img_path):
    plt.figure(figsize=(15, 7))
    plt.subplot(121)
    plt.title('train loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    loss_x = list(range(len(loss_history)))
    plt.plot(loss_x, loss_history)

    plt.subplot(122)
    plt.title('train accuracy')
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    acc_x = list(range(len(acc_history)))
    plt.plot(acc_x, acc_history)
    plt.savefig(img_path)


def add_handles(model, option, trainer, evaluator, train_loader, val_loader, optimizer):
    model_summary = summary(model, input_size=(option['seq_length'], option['input_size']),
                            batch_size=option['train_batch_size'])
    print(model_summary)
    pbar = ProgressBar()
    pbar.attach(trainer, output_transform=lambda loss_acc: {'loss': loss_acc[0],
                                                            'accuracy': loss_acc[1]})

    option['report_path'] = os.path.join(option['model_folder'], 'report.md')
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
        # BUG: 如果模型被earlystop terminal，则不会保存模型
        # BUG在测试中无法重现
        f = h5py.File(os.path.join(option['model_folder'], 'history.h5'), 'w')
        f.create_dataset('loss_history', data=loss_log)
        f.close()
        report_content = {
            'model_name': option['model'].upper(),
            'hyperparameter': option,
            'model_summary': model_summary,
            'log': report_log,
            'evaluation': 'evaluation result'
        }
        history_image_path = os.path.join(option['model_folder'], 'history.png')
        _plot_loss_history(loss_log, acc_log, history_image_path)
        store_report(report_content, option['report_path'])
        torch.save(model, os.path.join(option['model_folder'], 'model.pkl'))

    # add a learning rate scheduler
    step_scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    scheduler = LRScheduler(step_scheduler, save_history=True)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)

    # add early stop, if val loss has not improved for patience validation epochs,
    # training will stop
    def score_function(engine):
        val_loss = engine.state.metrics['loss']
        return -val_loss

    handler = EarlyStopping(patience=option['stop_patience'], score_function=score_function, trainer=trainer)
    # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
    evaluator.add_event_handler(Events.COMPLETED, handler)

    print('start train the model: {}'.format(option['model']))
    trainer.run(train_loader, max_epochs=option['epoch'])
