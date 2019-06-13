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
@click.argument('model', type=click.Choice(['lstm', 'seq2seq', 'mlp', 'cnn', 'c3d']))
@click.option('--suffix', type=str,  help='model name suffix')
@click.option('--sub_folder', type=str, help='sub-folder for saving checkpoints files and tensorboard log')
@click.option('--gesture_num', type=int, help='gesture number of classifier')
@click.option('--epoch', default=60, help='maximum epoch')
@click.option('--train_batch_size', default=128, help='')
@click.option('--valid_batch_size', default=512, help='')
@click.option('--lr', default=0.001, help='learning rate')
@click.option('--lr_step', type=int, help='learning rate decay step size')
@click.option('--stop_patience', type=int, help='early stop patience')
def main(**args):
    module_name = 'emg.models.{}'.format(args['model'])
    model = importlib.import_module(module_name)
    default_name = args['model'] + '-{}'.format(args['suffix'])
    args['name'] = default_name
    model.main(args)


if __name__ == '__main__':
    main()
