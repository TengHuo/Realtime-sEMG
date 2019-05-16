#!/miniconda3/envs/py36/bin/python
# -*- coding: UTF-8 -*-
# app.py
# @Time     : 15/May/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
# TODO: 整合lstm和seq2seq模型的代码到torch.py，复用框架代码，将模型代码分离
# 用args库改为命令行参数式训练，用.sh文件保存训练参数


import click
import importlib


@click.command()
@click.option('--model', default='seq2seq', help='Model name', type=click.Choice(['lstm', 'seq2seq']))
@click.option('--gesture_num', default=8, help='')
@click.option('--lr', default=0.01, help='Learning rate')
@click.option('--epoch', default=100, help='Train epoch')
@click.option('--train_batch_size', default=256, help='')
@click.option('--val_batch_size', default=1024, help='')
@click.option('--stop_patience', default=5, help='')
def main(**args):
    model_name = 'emg.models.{}'.format(args['model'])
    model = importlib.import_module(model_name)
    model.run(args)


if __name__ == '__main__':
    main()
