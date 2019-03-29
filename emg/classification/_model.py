# -*- coding: UTF-8 -*-
# _model.py
# @Time     : 22/Mar/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
# basic model class for capg gesture classification
# it can recognize 8-20 gestures according output_size setting


import os
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


class CapgModel(object):
    """basic model class for capg classifications
    """
    def __init__(self, model_name, batch_size, epoch, output_size):
        # init model parameters
        self.batch_size = batch_size
        self.epoch = epoch
        self.__model = None
        self.output_size = output_size
        model_folder_path, self.files_path = _generate_files_path(model_name, output_size)

        # create a folder for this model
        if not os.path.isdir(model_folder_path):
            os.makedirs(model_folder_path)

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        K.set_session(session)  # set this TensorFlow session as the default session for Keras

    def load_model(self, model_configure):
        model = model_configure()

        if os.path.exists(self.files_path['weights_file']):
            # model exist, load weights
            print('model exist! load weights from h5 file!')
            model.load_weights(self.files_path['weights_file'], by_name=True)
        elif 'trained_weights' in self.files_path.keys():
            # model doesn't exist, check if there previous model trained
            if os.path.exists(self.files_path['trained_weights']):
                print('pretrained model exist! load weights for previous layers!')
                model.load_weights(self.files_path['trained_weights'], by_name=True)

        output_layer_name = 'output_{}'.format(self.output_size)
        model.add(Dense(self.output_size, activation='softmax',
                        name=output_layer_name))
        self.__model = model

    def save_model(self):
        """save model and weights to h5 files
        """
        self.__model.save(self.files_path['model_file'])
        self.__model.save_weights(self.files_path['weights_file'])

    def compile_model(self, optimizer=None):
        if optimizer is None:
            optimizer = SGD(lr=0.01, momentum=0.9)

        self.__model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return self.__model.summary()

    def train_model(self, x_train, y_train, val_split=0.01, callbacks=None):
        if callbacks is None:
            callbacks = list()

        # TODO: add early stop and model checkpoint callbacks
        logger = CSVLogger(self.files_path['log_file'])
        callbacks.append(logger)
        callbacks.append(_lr_tuner_configure(self.epoch))

        val_index = np.random.choice(x_train.shape[0], int(x_train.shape[0]*val_split), replace=False)
        train_index = np.delete(np.arange(x_train.shape[0]), val_index)
        x_val = x_train[val_index]
        y_val = y_train[val_index]
        x_train = x_train[train_index]
        y_train = y_train[train_index]

        history = self.__model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epoch,
                                 validation_data=(x_val, y_val), callbacks=callbacks)
        self.save_model()
        return history.history

    def evaluate_model(self, x_test, y_test):
        score = self.__model.evaluate(x=x_test, y=y_test, batch_size=self.batch_size)
        return score


# callbacks settings
def _lr_tuner_configure(epoch):
    """ set learning rate scheduler
    """
    step1 = epoch * (1/3)
    step2 = epoch * (2/3)

    def learning_rate_turner(epoch):
        lr = 0.1
        if step1 <= epoch < step2:
            lr = 0.01
        elif epoch >= step2:
            lr = 0.001
        return lr

    return LearningRateScheduler(learning_rate_turner)


def _generate_files_path(model_name, out_size):
    """return the store path of this model
    """
    files_path = dict()
    root_path = os.path.join(os.sep, *os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-2])
    model_folder = os.path.join(root_path, 'models', model_name, '{}'.format(out_size))

    # generate path
    file_name = 'weights-{}.h5'.format(out_size)
    files_path['weights_file'] = os.path.join(model_folder, file_name)

    file_name = 'model-{}.h5'.format(out_size)
    files_path['model_file'] = os.path.join(model_folder, file_name)

    file_name = 'history-{}'.format(out_size)
    files_path['history'] = os.path.join(model_folder, file_name)

    file_name = 'train-{}.log'.format(out_size)
    files_path['log_file'] = os.path.join(model_folder, file_name)

    if out_size > 8:
        file_name = 'weights-{}.h5'.format(out_size-1)
        files_path['trained_weights'] = os.path.join(root_path, 'models', model_name,
                                                        '{}'.format(out_size-1), file_name)

    return model_folder, files_path


if __name__ == "__main__":
    # test file path generator
    files_path = _generate_files_path('test', 10)
    print(files_path)
