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

# TODO: 先走通一个MLP和一个CNN

import os

# 检查数据是否存在
data_file_path = ''
if not os.path.isfile(data_file_path):
    test = load_capg_all(LoadMode.flat)
    train, test = capg_split_train_test(test, test_size=0.1)
    save_capg_to_h5(train, test, data_file_path)
else:
    train, test = load_capg_from_h5(data_file_path)

x_train, y_train = prepare_data(train)
x_test, y_test = prepare_data(test)

# 训练MLP的8 gesture classification模型
# 固定前几层weights，替换最后一层output为9-20个输出再分别训练
# 测试模型，保存测试结果
model = Capg_MLP()
model.train()




# TODO:
# 依次训练CNN，LSTM，ConvLSTM（待修改为论文的模型）的8 gesture classification模型
# 修改output层，训练9-20的classification
# 测试模型

