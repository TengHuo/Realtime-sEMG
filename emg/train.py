#!/miniconda3/envs/py36/bin/python
# -*- coding: UTF-8 -*-
# train.py
# @Time     : 15/May/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#


import click
import importlib


@click.command()
@click.option('--model', default='seq2seq', help='Model name',
              type=click.Choice(['lstm', 'seq2seq', 'mlp', 'cnn', 'c3d']))
@click.option('--suffix', default='default', help='')
@click.option('--sub_folder', default='default', help='')
@click.option('--gesture_num', default=8, help='')
@click.option('--lr', default=0.001, help='learning rate')
@click.option('--lr_step', default=10, help='learning rate decay step size')
@click.option('--epoch', default=100, help='maximum epoch')
@click.option('--train_batch_size', default=256, help='')
@click.option('--valid_batch_size', default=1024, help='')
@click.option('--stop_patience', default=5, help='')
@click.option('--log_interval', default=100, type=int, help='')
def main(**args):
    module_name = 'emg.models.{}'.format(args['model'])
    model = importlib.import_module(module_name)
    default_name = args['model'] + '-{}'.format(args['suffix'])
    args['name'] = default_name
    model.main(args)


if __name__ == '__main__':
    main()