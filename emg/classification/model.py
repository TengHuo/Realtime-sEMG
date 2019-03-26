# -*- coding: UTF-8 -*-
# model.py
# @Time     : 22/Mar/2019
# @Auther   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
# 模型的父类，设置模型的基本参数，
# 生成文件名，在cache中载入和保存模型


import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras import utils
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras import backend as K

MODEL_FOLDER = './models/'

class EMG_model(object):
    '''
    设置模型的基本参数：batch，epoch， 

    模型保存和载入
    '''
    def __init__(self, model_name):
        # 初始化模型参数
        self.batch_size = 128
        self.epoch = 60
        self.model = None
        self.history = None
        self.files_path = self._generate_files_name(model_name)


    def _generate_files_name(self, model_name):
        # 返回模型相关的文件路径
        return {'weight_file': 'name', 'model_file': 'name', 'acc_figure': 'name', 'loss_figure': 'name', 'history': 'name'}

    def load_model_history(self):
        # 载入模型训练history
        pass

    def load_model(self, optimizer, model_configure):
        model = model_configure()

        if os.path.exists(self.files_path['weight_file']): # model exist
            model.load_weights(self.files_path['weight_file'], by_name=True)

        self.model = model
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model.summary()


    def save_model(self):
        '''save model to files
        '''
        pass

    def train_model(self, x_train, y_train, val_split=0.01, callbacks=None):
        self.history = self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epoch,
                    validation_split=val_split, callbacks=callbacks)
        self.save_model()
        return self.history

    def lr_tuner_configure(self, lr_list):
        '''用python的装饰器包装一个learning rate scheduler
        '''
        lr = 0.1
        if 30 <= epoch < 60:
            lr = 0.01
        elif epoch >= 60:
            lr = 0.001
        return lr