# -*- coding: UTF-8 -*-
# net.py
# @Time     : 24/May/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
#


import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from emg.models.train_manager import Manager


hyperparameters = {
    'input_size': (1, 28, 28),
    'seq_result': False,
    'frame_input': True
}


class Net(nn.Module):
    def __init__(self, gesture_num: int):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(20, 50, 3, stride=1, padding=1)
        self.conv1.weight.data.normal_()
        self.fc1 = nn.Linear(2450, 500)
        self.fc2 = nn.Linear(500, gesture_num)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 2450)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# for test mnist data
def transform_fun(img):
    to_tensor = transforms.ToTensor()
    img = to_tensor(img)
    # img = img[0, :16, :8]
    # img = img.view(1, 16, 8)
    # print(img.size())
    return img


# for test mnist data
def mnist_loader(args):
    train_data = DataLoader(dataset=datasets.MNIST(root='../../cache/',
                                                   transform=transform_fun,
                                                   train=True,
                                                   download=True),
                            batch_size=args['train_batch_size'],
                            shuffle=True)
    test_data = DataLoader(dataset=datasets.MNIST(root='../../cache/',
                                                  transform=transform_fun,
                                                  train=False),
                           batch_size=args['val_batch_size'],
                           shuffle=False)

    return train_data, test_data


def main(train_args):
    # 1. 设置好optimizer
    # 2. 定义好model
    args = {**train_args, **hyperparameters}
    model = Net(args['gesture_num'])
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9, weight_decay=0.01)

    manager = Manager(args, model, mnist_loader)
    manager.compile(optimizer)
    manager.summary()
    manager.start_train()
    # manager.test()  # TODO
    manager.finish()


if __name__ == "__main__":
    test_args = {
        'model': 'net',
        'gesture_num': 10,
        'lr': 0.001,
        'lr_step': 5,
        'epoch': 10,
        'train_batch_size': 256,
        'val_batch_size': 1024,
        'stop_patience': 5,
        'log_interval': 100,
        'test': True
    }

    main(test_args)
