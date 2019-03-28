# -*- coding: UTF-8 -*-
# _model.py
# @Time     : 22/Mar/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
# 模型的父类，设置模型的基本参数，
# 生成文件名，在cache中载入和保存模型


import os

from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


class CapgModel(object):
    """basic model class for capg classifications
    """
    def __init__(self, model_name, batch_size=128, epoch=60, output_size=8):
        # 初始化模型参数
        self.batch_size = batch_size
        self.epoch = epoch
        self.model = None
        self.output_size = output_size
        self.files_path = self.__generate_files_path(model_name)

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        K.set_session(session)  # set this TensorFlow session as the default session for Keras

    def __generate_files_path(self, model_name):
        # 返回模型相关的文件路径
        files_path = dict()
        root_path = os.path.join(os.sep, *os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-2])
        model_folder = os.path.join(root_path, 'models', model_name, '{}'.format(self.output_size))

        # create a folder for storing models
        if not os.path.isdir(model_folder):
            os.makedirs(model_folder)

        file_name = 'weights-{}.h5'.format(self.output_size)
        files_path['weights_file'] = os.path.join(model_folder, file_name)

        file_name = 'model-{}.h5'.format(self.output_size)
        files_path['model_file'] = os.path.join(model_folder, file_name)

        file_name = 'history-{}'.format(self.output_size)
        files_path['history'] = os.path.join(model_folder, file_name)

        file_name = 'train-{}.log'.format(self.output_size)
        files_path['log_file'] = os.path.join(model_folder, file_name)

        if self.output_size > 8:
            file_name = 'weights-{}.h5'.format(self.output_size-1)
            files_path['trained_weights'] = os.path.join(root_path, 'models', model_name,
                                                         '{}'.format(self.output_size-1), file_name)

        return files_path

    def load_model(self, model_configure):
        model = model_configure()

        if os.path.exists(self.files_path['weights_file']):
            # model exist, load weights
            model.load_weights(self.files_path['weights_file'], by_name=True)
        elif 'trained_weights' in self.files_path.keys():
            # model doesn't exist, check if there previous model trained
            if os.path.exists(self.files_path['trained_weights']):
                model.load_weights(self.files_path['trained_weights'], by_name=True)

        output_layer_name = 'output_{}'.format(self.output_size)
        model.add(Dense(self.output_size, activation='softmax',
                        name=output_layer_name))
        return model

    def save_model(self):
        """save model and weights to h5 files
        """
        self.model.save(self.files_path['model_file'])
        self.model.save_weights(self.files_path['weights_file'])

    @property
    def train_history(self):
        return self.__history

    @train_history.setter
    def train_history(self, history):
        self.__history = history

    def compile_model(self, optimizer=None):
        if optimizer is None:
            optimizer = SGD(lr=0.01, momentum=0.9)

        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return self.model.summary()

    def train_model(self, x_train, y_train, val_split=0.01, callbacks=None):
        if callbacks is None:
            callbacks = list()
        logger = CSVLogger(self.files_path['log_file'])
        callbacks.append(logger)
        callbacks.append(self.lr_tuner_configure())

        history = self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epoch,
                                 validation_split=val_split, callbacks=callbacks)
        self.save_model()
        self.train_history = history.history
        return self.train_history

    def evaluate_model(self, x_test, y_test):
        score = self.model.evaluate(x=x_test, y=y_test, batch_size=self.batch_size)
        return score

    def set_output_size(self, size=8):
        self.output_size = size

    def add_output_layer(self, out_size=8):
        self.output_size = out_size
        self.model.layers.pop()
        layer_name = 'output_{}'.format(out_size)
        return Dense(out_size, activation='softmax', name=layer_name)

    def lr_tuner_configure(self):
        """ set learning rate scheduler
        """
        step1 = self.epoch * (1/3)
        step2 = self.epoch * (2/3)

        def learning_rate_turner(epoch):
            lr = 0.1
            if step1 <= epoch < step2:
                lr = 0.01
            elif epoch >= step2:
                lr = 0.001
            return lr

        return LearningRateScheduler(learning_rate_turner)
