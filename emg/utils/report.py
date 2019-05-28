# -*- coding: UTF-8 -*-
# report-generator.py
# @Time     : 17/May/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
# 根据template.md生成name-date-time.md

import time
import os
import matplotlib.pyplot as plt
import numpy as np


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
    template = template.replace('{{date}}', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    template = template.replace('{{hyperparameters}}', args_str)
    template = template.replace('{{summary}}', content['model_summary'])
    template = template.replace('{{log}}', '\n'.join(content['log']))
    template = template.replace('{{image}}', './history.png')
    generated_report = template.replace('{{evaluation}}', content['evaluation'])
    return generated_report


def save_history_figures(train_array: np.ndarray, eval_array: np.ndarray, img_path):
    """ history format: [[iteration index, loss, accuracy]] """
    plt.figure(figsize=(15, 7))
    plt.subplot(121)
    plt.title('loss history')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.plot(train_array[:, 0], train_array[:, 1], )
    plt.plot(eval_array[:, 0], eval_array[:, 1])
    plt.legend(['Train', 'Evaluation'], loc='upper left')

    plt.subplot(122)
    plt.title('accuracy history')
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.plot(train_array[:, 0], train_array[:, 2])
    plt.plot(eval_array[:, 0], eval_array[:, 2])
    plt.legend(['Train', 'Evaluation'], loc='upper left')

    plt.savefig(img_path)


def save_report(report_content: dict, file_path: str) -> bool:
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

    save_report(test_content, '../../outputs/test.md')
