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
    template = template.replace('{{name}}', content['model_name'])
    template = template.replace('{{date}}', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    template = template.replace('{{hyperparameters}}', str(content['hyperparameter']))
    template = template.replace('{{summary}}', content['model_summary'])
    template = template.replace('{{log}}', '\n'.join(content['log']))
    template = template.replace('{{image}}', './history.png')
    generated_report = template.replace('{{evaluation}}', content['evaluation'])
    return generated_report


def store_report(report_content: dict, file_path: str) -> bool:
    template = _read_template()
    new_report = _generate(report_content, template)
    try:
        with open(file_path, 'w') as f:
            f.write(new_report)
        return True
    except IOError:
        print('test error')


if __name__ == '__main__':
    test_content = {
        'model_name': 'test',
        'hyperparameter': {'test': 1, 'test2': 2},
        'model_summary': 'test',
        'log': "Training Results - Avg accuracy: 0.10 Avg loss: 0.10\n",
        'history_img_path': './history.png',
        'evaluation': 'evaluation result'
    }

    store_report(test_content, '../../outputs/test.md')
