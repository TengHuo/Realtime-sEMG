# -*- coding: UTF-8 -*-
# c3d.py
# @Time     : 06/Jun/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
#


import torch
import torch.nn as nn
import torch.nn.functional as F

from emg.models.base import EMGClassifier
from emg.data_loader.capg_data import CapgDataset


hyperparameters = {
    'input_size': (10, 16, 8),
    'hidden_size': 256
}


class C3D(nn.Module):
    def __init__(self, output_size):
        super(C3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        #
        # self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(1024, 512)
        self.fc7 = nn.Linear(512, 512)
        self.fc8 = nn.Linear(512, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = self.pool3(x)

        # x = F.relu(self.conv4a(x))
        # x = F.relu(self.conv4b(x))
        # x = self.pool4(x)

        # x = F.relu(self.conv5a(x))
        # x = F.relu(self.conv5b(x))
        # x = self.pool5(x)

        x = x.view(-1, 1024)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, p=0.2)

        x = F.relu(self.fc7(x))
        x = F.dropout(x, p=0.2)
        output = self.fc8(x)
        return output


def main(train_args, TEST_MODE=False):
    # 1. 设置好optimizer
    # 2. 定义好model
    args = {**train_args, **hyperparameters}
    model = C3D(args['gesture_num'])
    name = args['model'] + '-' + str(args['gesture_num'])
    sub_folder = 'test'

    from emg.utils import config_tensorboard
    tensorboard_cb = config_tensorboard(name, sub_folder, model, (1, 1, 10, 16, 8))

    # from emg.utils.lr_scheduler import DecayLR
    # lr_callback = DecayLR(start_lr=0.001, gamma=0.1, step_size=12)

    train_set = CapgDataset(gesture=args['gesture_num'],
                            sequence_len=10,
                            frame_x=True,
                            test_mode=TEST_MODE,
                            train=True)

    net = EMGClassifier(module=model,
                        model_name=name,
                        sub_folder=sub_folder,
                        hyperparamters=args,
                        optimizer=torch.optim.Adam,
                        max_epochs=args['epoch'],
                        lr=args['lr'],
                        dataset=train_set,
                        callbacks=[tensorboard_cb])

    net.fit_with_dataset()

    # test_set = CapgDataset(gesture=args['gesture_num'],
    #                        sequence_len=1,
    #                        test_mode=TEST_MODE,
    #                        train=False)
    #
    # avg_score = net.test_model(test_set)
    # print('test accuracy: {:.4f}'.format(avg_score))


if __name__ == "__main__":
    test_args = {
        'model': 'c3d',
        'gesture_num': 8,
        'lr': 0.001,
        'lr_step': 5,
        'epoch': 60,
        'train_batch_size': 512,
        'valid_batch_size': 1024,
        'stop_patience': 5,
        'log_interval': 100
    }

    main(test_args, TEST_MODE=False)
