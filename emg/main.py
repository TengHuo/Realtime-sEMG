# -*- coding: UTF-8 -*-
# main.py
# @Time     : 13/Dec/2018
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
# 在main.py中训练模型并保存参数和图表
# 在jupyter中展示和比较模型结果

# TODO: 将代码整合到app中
# 使用命令行神器 Click
# 参考： http://funhacks.net/2016/12/20/click/
# https://isudox.com/2016/09/03/learning-python-package-click/
# https://github.com/TengHuo/srep/blob/master/sigr/app.py

from emg.utils import *
from emg.models import MLP, CNN
import os

# load config setting
try:
    from emg.config_default import configs
except ImportError:
    configs = {'test': False}

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

for gesture_amount in range(20, 21):
    x_train, y_train = prepare_data(train, required_gestures=gesture_amount,
                                    mode=LoadMode.flat_frame)
    x_test, y_test = prepare_data(test, required_gestures=gesture_amount,
                                  mode=LoadMode.flat_frame)

    # mlp = CapgMLP('MLP', epoch=30, output_size=gesture_amount)
    # model_summary = mlp.build_model()
    # print(model_summary)

    # history = mlp.train(x_train, y_train, val_split=0.01)
    # history['score'] = mlp.evaluate_model(x_test, y_test)
    # save_history(history, mlp.files_path['history'])
    # print(history)

    # For TEST
    if configs['test']:
        x_train = x_train[0:1280]
        y_train = y_train[0:1280]
        x_test = x_test[0:1280]
        y_test = y_test[0:1280]
        cnn = CNN('CNN', epoch=1, output_size=gesture_amount)
    else:
        cnn = CNN('CNN', epoch=60, output_size=gesture_amount)

    model_summary = cnn.build_model()
    print(model_summary)

    history = cnn.train(x_train, y_train, val_split=0.01)
    history['score'] = cnn.evaluate_model(x_test, y_test)
    save_history(history, cnn.files_path['history'])
    print(history)
    # TODO:每次训练完清空内存

    # for test
    if configs['test']:
        break
