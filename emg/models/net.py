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

from torch.utils.tensorboard import SummaryWriter

from emg.models.base import EMGClassifier
from emg.utils import TensorboardCallback, generate_folder, init_parameters


hyperparameters = {
    'input_size': (1, 28, 28),
    'seq_result': False,
    'frame_input': True
}


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(20, 50, 3, stride=1, padding=1)
        self.conv1.weight.data.normal_()
        self.fc1 = nn.Linear(2450, 500)
        self.fc2 = nn.Linear(500, 10)

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

    train_set = datasets.MNIST(root='../../data/',
                               transform=transform_fun,
                               train=True,
                               download=True)

    x = train_set.data
    x = x.to(torch.float32)
    x /= 255.0
    x = x.view(x.size(0), 1, 28, 28)
    y = train_set.targets

    # model = Net()
    # f_name = args['model'] + '-' + str(args['gesture_num']) + '-no_init'

    # tb_dir = generate_folder(root_folder='tensorboard', folder_name=f_name, sub_folder='3fc-2bn')
    # writer = SummaryWriter(tb_dir)
    # dummpy_input = torch.ones((1, 1, 28, 28), dtype=torch.float, requires_grad=True)
    # writer.add_graph(model, input_to_model=dummpy_input)

    # tensorboard_cb = TensorboardCallback(writer)
    # net = EMGClassifier(module=model, model_name=f_name,
    #                     hyperparamters=args,
    #                     lr=args['lr'],
    #                     batch_size=args['train_batch_size'],
    #                     continue_train=False,
    #                     stop_patience=args['stop_patience'],
    #                     max_epochs=args['epoch'],
    #                     optimizer=torch.optim.Adam,
    #                     callbacks=[tensorboard_cb])

    from emg.models.test import NeuralNetClassifier
    # model = Net()
    # model.apply(init_parameters)

    from skorch.dataset import CVSplit

    net = NeuralNetClassifier(module=Net,
                              criterion=nn.CrossEntropyLoss,
                              optimizer=torch.optim.Adam,
                              train_split=CVSplit(10),
                              max_epochs=args['epoch'],
                              device='cuda')
    print('start train')
    net.fit(x, y)


if __name__ == "__main__":
    test_args = {
        'model': 'net',
        'gesture_num': 10,
        'lr': 0.001,
        'lr_step': 5,
        'epoch': 30,
        'train_batch_size': 128,
        'val_batch_size': 1024,
        'stop_patience': 8,
        'log_interval': 100,
        'test': True
    }

    main(test_args)
