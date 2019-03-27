# -*- coding: UTF-8 -*-
# _model.py
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
import h5py
import pickle

from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


class CapgModel(object):
    '''
    设置模型的基本参数：batch，epoch， 

    模型保存和载入
    '''
    def __init__(self, model_name, batch_size=128, epoch=60):
        # 初始化模型参数
        self.batch_size = batch_size
        self.epoch = epoch
        self.model = None
        self.files_path = self._generate_files_path(model_name)

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        K.set_session(session)  # set this TensorFlow session as the default session for Keras


    def _generate_files_path(self, model_name):
        # 返回模型相关的文件路径
        file_names = dict()
        root_path = os.path.join(os.sep, *os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-2])
        model_folder = os.path.join(root_path, 'models', model_name)

        # create a folder for storing models
        if not os.path.isdir(model_folder):
            os.mkdir(model_folder)

        file_names['weights_file'] = os.path.join(model_folder, 'weights.h5')
        file_names['model_file'] = os.path.join(model_folder, 'model.h5')
        file_names['history'] = os.path.join(model_folder, 'history')

        return file_names

    def load_model(self, model_configure):
        model = model_configure()

        if os.path.exists(self.files_path['weights_file']): # model exist
            model.load_weights(self.files_path['weights_file'], by_name=True)

        return model

    def save_model(self):
        '''save model and weights to h5 files
        '''
        self.model.save(self.files_path['model_file'])
        self.model.save_weights(self.files_path['weights_file'])

    @property
    def train_history(self):
        # 载入模型训练history
        if self.__history is None:
            with open(self.files_path['history'], 'r') as history_file:
                self.__history = pickle.load(history_file)
        return self.__history

    @train_history.setter
    def train_history(self, history):
        self.__history = history
        with open(self.files_path['history'], 'wb') as history_file:
            pickle.dump(history, history_file)


    def compile_model(self, optimizer=None):
        if optimizer is None:
            optimizer = SGD(lr=0.01, momentum=0.9)

        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model.summary()


    def train_model(self, x_train, y_train, val_split=0.01, callbacks=None):
        history = self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epoch,
                    validation_split=val_split, callbacks=callbacks)
        self.save_model()
        self.train_history = history.history
        return self.train_history

    def lr_tuner_configure(self, lr_list):
        '''TODO: 用python的装饰器包装一个learning rate scheduler
        '''
        pass
        # lr = 0.1
        # if 30 <= epoch < 60:
        #     lr = 0.01
        # elif epoch >= 60:
        #     lr = 0.001
        # return lr

    # TODO:
    def tensorboard_configure(self):
        pass

    # TODO:
    def log_configure(self):
        pass


if __name__ == "__main__":
    test = CapgModel('test')
    print(test.files_path)