# -*- coding: UTF-8 -*-
# capg_mlp.py
# @Time     : 22/Mar/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
#

# 需要解决的问题：
# 1. 如何导入同一目录下的类，python的包管理
# 2. 相对路径问题


from ._model import CapgModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation

from tensorflow.keras.optimizers import Adam


def _model_configure():
    model = Sequential()

    model.add(Dense(256, input_dim=128, name='dense_1'))
    model.add(BatchNormalization(momentum=0.9, name='bn_1'))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.2, name='dp_1'))

    model.add(Dense(256, name='dense_2'))
    model.add(BatchNormalization(momentum=0.9, name='bn_2'))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.2, name='dp_2'))

    model.add(Dense(256, name='dense_3'))
    model.add(BatchNormalization(momentum=0.9, name='bn_3'))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.2, name='dp_3'))

    return model


# my_optimizer = Adam()
_my_optimizer = None


class CapgMLP(CapgModel):
    def __init__(self, model_name='MLP', batch_size=128, epoch=60, output_size=8):
        CapgModel.__init__(self, model_name, batch_size, epoch, output_size)

    def build_mlp(self):
        self.model = self.load_model(_model_configure)
        summary = self.compile_model(_my_optimizer)
        print(summary)
        return summary

    def train(self, x_train, y_train, val_split=0.01):
        self.train_model(x_train, y_train, val_split)


if __name__ == "__main__":
    # TODO: 测试模型
    test = CapgMLP('test')
    print(test.files_path)


