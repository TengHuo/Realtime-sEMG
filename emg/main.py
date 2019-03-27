# -*- coding: UTF-8 -*-
# main.py
# @Time     : 13/Dec/2018
# @Auther   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
# 在main.py中训练模型并保存参数和图表
# 在jupyter中展示和比较模型结果

from utils import *
from classification import CapgMLP
import os
import matplotlib.pyplot as plt

# 检查数据是否存在
root_path = os.path.join(os.sep, *os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-1])
data_file_path = os.path.join(root_path, 'cache', 'capg.h5')
if not os.path.isfile(data_file_path):
    print('data not exist, create a new h5 file')
    capg_data = load_capg_all()
    train, test = capg_train_test_split(capg_data, test_size=0.1)
    save_capg_to_h5(train, test, data_file_path)
else:
    print('data exist, load data from the file')
    train, test = load_capg_from_h5(data_file_path)

print('data load complete, start train model')
# 训练MLP的8 gesture classification模型
# 固定前几层weights，替换最后一层output为9-20个输出再分别训练
# 测试模型，保存测试结果

for gesture_amount in range(8, 21):
    x_train, y_train = prepare_data(train, required_gestures=gesture_amount,
                                    mode=LoadMode.flat)

    mlp = CapgMLP('MLP', epoch=20, output_size=gesture_amount)
    mlp.build_mlp()
    mlp.train(x_train, y_train, val_split=0.01)
    history = mlp.train_history
    print(history)





# # TODO:
# # 依次训练CNN，LSTM，ConvLSTM（待修改为论文的模型）的8 gesture classification模型
# # 修改output层，训练9-20的classification
# # 测试模型

