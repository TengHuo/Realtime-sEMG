# -*- coding: UTF-8 -*-
# capg_cnn.py
# @Time     : 22/Mar/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
#


from ._model import CapgModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LocallyConnected2D, MaxPooling2D, Activation
from tensorflow.keras.layers import BatchNormalization, Dropout, Dense, Flatten
from tensorflow.keras.optimizers import Adam


def _model_configure():
    model = Sequential()

    model.add(BatchNormalization(input_shape=[16, 8, 1], momentum=0.9, name='input'))

    model.add(Conv2D(64, (3, 3), use_bias=False,
                    padding='same', strides=(1, 1), name='cov_1'))
    model.add(BatchNormalization(momentum=0.9, name='bn_1'))
    model.add(Activation('relu', name='ac_1'))

    model.add(Conv2D(64, (3, 3), use_bias=False,
                    padding='same', strides=(1, 1), name='cov_2'))
    model.add(BatchNormalization(momentum=0.9, name='bn_2'))
    model.add(Activation('relu', name='ac_2'))

    model.add(LocallyConnected2D(64, (1, 1), use_bias=False, name='lc_1'))
    model.add(BatchNormalization(momentum=0.9, name='bn_3'))
    model.add(Activation('relu', name='ac_3'))

    model.add(LocallyConnected2D(64, (1, 1), use_bias=False, name='lc_2'))
    model.add(BatchNormalization(momentum=0.9, name='bn_4'))
    model.add(Activation('relu', name='ac_4'))
    model.add(Dropout(0.5, name='dp_1'))

    model.add(Flatten(name='flat'))
    model.add(Dense(units=512, use_bias=False, name='den_1'))
    model.add(BatchNormalization(momentum=0.9, name='bn_5'))
    model.add(Activation('relu', name='ac_5'))
    model.add(Dropout(0.5, name='dp_2'))

    model.add(Dense(units=512, use_bias=False, name='den_2'))
    model.add(BatchNormalization(momentum=0.9, name='bn_6'))
    model.add(Activation('relu', name='ac_6'))
    model.add(Dropout(0.5, name='dp_3'))

    model.add(Dense(units=128, use_bias=False, name='den_3'))
    model.add(BatchNormalization(momentum=0.9, name='bn_7'))
    model.add(Activation('relu', name='ac_7'))

    return model


# my_optimizer = Adam()
_my_optimizer = None


class CapgCNN(CapgModel):
    def __init__(self, model_name='CNN', batch_size=128, epoch=60, output_size=8):
        CapgModel.__init__(self, model_name, batch_size, epoch, output_size)

    def build_model(self):
        self.load_model(_model_configure)
        summary = self.compile_model(_my_optimizer)
        return summary

    def train(self, x_train, y_train, val_split=0.01):
        return self.train_model(x_train, y_train, val_split)
