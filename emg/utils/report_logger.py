# -*- coding: UTF-8 -*-
# logger.py
# @Time     : 17/May/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
#

import sys
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from skorch.callbacks import Callback


class ReportLog(Callback):
    """Print useful information from the model's history as a table.

    By default, ``PrintLog`` prints everything from the history except
    for ``'batches'``.

    To determine the best loss, ``PrintLog`` looks for keys that end on
    ``'_best'`` and associates them with the corresponding loss. E.g.,
    ``'train_loss_best'`` will be matched with ``'train_loss'``. The
    ``Scoring`` callback takes care of creating those entries, which is
    why ``PrintLog`` works best in conjunction with that callback.

    ``PrintLog`` treats keys with the ``'event_'`` prefix in a special
    way. They are assumed to contain information about occasionally
    occuring events. The ``False`` or ``None`` entries (indicating
    that an event did not occur) are not printed, resulting in empty
    cells in the table, and ``True`` entries are printed with ``+``
    symbol. ``PrintLog`` groups all event columns together and pushes
    them to the right, just before the ``'dur'`` column.

    *Note*: ``PrintLog`` will not result in good outputs if the number
    of columns varies between epochs, e.g. if the valid loss is only
    present on every other epoch.

    Parameters
    ----------
    sink : callable (default=print)
      The target that the output string is sent to. By default, the
      output is printed to stdout, but the sink could also be a
      logger, etc.

    """
    def __init__(self, sink=print):
        self.sink = sink
        self.__log_report = []
        self.__start_time = ''

    def _sink(self, text, verbose):
        if (self.sink is not print) or verbose:
            self.sink(text)

    def on_train_begin(self, net,
                       X=None, y=None, **kwargs):
        """Called at the beginning of training."""
        # generate a new report
        self.__log_report = []
        self.__start_time = datetime.now().strftime('%m-%d-%H-%M-%S')

    def on_train_end(self, net, X=None, y=None, **kwargs):
        """Called at the end of training."""
        # save report to disk
        report_content = {
            'datetime': self.__start_time,
            'model_name': net.model_name.upper(),
            'model_summary': repr(net.module),
            'hyperparameter': net.hyperparamters,
            'log': self.__log_report,
            'evaluation': 'evaluation result'
        }
        report_path = os.path.join(net.model_path, 'report.md')
        _save_report(report_content, report_path)
        # generate a firgure of loss and accuracy
        history = net.history
        train_history = []
        valid_history = []
        batch_step = 0
        for epoch in range(len(history)):
            history_batch = history[epoch, 'batches']
            for batch_info in history_batch:
                batch_step += 1
                if 'train_loss' in batch_info and \
                        batch_step % net.hyperparamters['log_interval'] == 1:
                    train_history.append([batch_step, batch_info['train_loss']])
            valid_history.append([batch_step, history[epoch, 'valid_loss'], history[epoch, 'valid_acc']])
        train_history = np.array(train_history)
        valid_history = np.array(valid_history)
        history_fig_path = os.path.join(net.model_path, 'history.png')
        _save_history_figures(train_history, valid_history, history_fig_path)

    def on_epoch_end(self, net, **kwargs):
        data = net.history[-1]
        train_loss = data['train_loss']
        val_loss = data['valid_loss']
        val_acc = data['valid_acc']
        log = 'Epoch {}, Train loss: {:.4f} - Valid loss: {:.4f} - Valid accuracy: {:.4f}' \
              .format(data['epoch'], train_loss, val_loss, val_acc)
        self.__log_report.append(log)
        print()
        self._sink(log, net.verbose)
        sys.stdout.flush()


def save_evaluation(report_folder: str, score: float):
    try:
        report_path = os.path.join(report_folder, 'report.md')
        with open(report_path, 'r+') as f:
            report = f.read()
            report = report.replace('{{evaluation}}', '{:.4f}'.format(score))
            f.write(report)
    except IOError as e:
        print('read file failure')
        print(e)


def _read_template() -> str:
    try:
        template_path = os.path.join(os.sep, *os.path.dirname(os.path.realpath(__file__)).split(os.sep))
        template_path = os.path.join(template_path, 'template.md')
        with open(template_path, 'r') as f:
            template = f.read()
        return template
    except IOError as e:
        print(e)
        print('read file failure')


def _generate(content: dict, template: str) -> str:
    args_str = ''
    for k, v in content['hyperparameter'].items():
        args_str += '- **{}**: {}\n'.format(k, v)

    template = template.replace('{{name}}', content['model_name'])
    template = template.replace('{{date}}', content['datetime'])
    template = template.replace('{{hyperparameters}}', args_str)
    template = template.replace('{{summary}}', content['model_summary'])
    template = template.replace('{{log}}', '\n'.join(content['log']))
    generated_report = template.replace('{{image}}', './history.png')
    # generated_report = template.replace('{{evaluation}}', content['evaluation'])
    return generated_report


def _save_history_figures(train_array: np.ndarray, valid_array: np.ndarray, img_path):
    """ history format: [[iteration index, loss, accuracy]] """
    plt.figure(figsize=(15, 7))
    plt.subplot(121)
    plt.title('loss history')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.plot(train_array[:, 0], train_array[:, 1])
    plt.plot(valid_array[:, 0], valid_array[:, 1])
    plt.legend(['Train', 'Evaluation'], loc='upper right')

    plt.subplot(122)
    plt.title('accuracy history')
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.plot(valid_array[:, 0], valid_array[:, 2])

    plt.savefig(img_path)


def _save_report(report_content: dict, file_path: str) -> bool:
    template = _read_template()
    new_report = _generate(report_content, template)
    try:
        with open(file_path, 'w') as f:
            f.write(new_report)
        return True
    except IOError:
        print('test error')


if __name__ == '__main__':
    test_arg1 = {'test': 1, 'test2': 2}
    test_arg2 = {'test3': '123', 'test4': True}
    test_arg = {**test_arg1, **test_arg2}

    test_content = {
        'model_name': 'test',
        'hyperparameter': test_arg,
        'model_summary': 'test',
        'log': "Training Results - Avg accuracy: 0.10 Avg loss: 0.10\n",
        'history_img_path': './history.png',
        'evaluation': 'evaluation result'
    }

    _save_report(test_content, '../../outputs/test.md')
