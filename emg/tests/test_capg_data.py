# -*- coding: UTF-8 -*-
# test_capg_data.py
# @Time     : 28/Mar/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
# TODO: add more test case


import os
import unittest
from emg.utils import *

class TestCapgData(unittest.TestCase):
    
    def setUp(self):
        now_path = os.path.join(os.sep, *os.path.dirname(os.path.realpath(__file__)).split(os.sep)[:-2])
        self.h5_file_path = os.path.join(now_path, 'cache', 'test.h5')

        test_data = load_capg_all()
        self.data = test_data
        return super().setUp()

    def tearDown(self):
        """delete test files
        """
        os.remove(self.h5_file_path)
        return super().tearDown()

    def test_capg_train_test_split(self):
        train, test = capg_train_test_split(self.data, test_size=0.1)
        self.assertEqual(train.keys(), list(range(20)))
        self.assertEqual(test.keys(), list(range(20)))

    def test_save_data(self):
        train, test = capg_train_test_split(self.data, test_size=0.1)
        save_capg_to_h5(train, test, self.h5_file_path)
    
    def test_load_data(self):
        train, test = load_capg_from_h5(self.h5_file_path)
        self.assertIs(train, dict)
        self.assertIs(test, dict)

    def test_prepare_data(self):
        # TODO: 把print修改为assert
        train, test = load_capg_from_h5(self.h5_file_path)

        x_train, y_train = prepare_data(train, mode=LoadMode.sequence)
        x_test, y_test = prepare_data(test, mode=LoadMode.sequence)
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)
        print()

        x_train, y_train = prepare_data(train, mode=LoadMode.flat)
        x_test, y_test = prepare_data(test, mode=LoadMode.flat)
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)
        print()

        x_train, y_train = prepare_data(train, mode=LoadMode.flat_frame)
        x_test, y_test = prepare_data(test, mode=LoadMode.flat_frame)
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)
        print()

        x_train, y_train = prepare_data(train, mode=LoadMode.sequence_frame)
        x_test, y_test = prepare_data(test, mode=LoadMode.sequence_frame)
        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)
        print()

        # test the function prepare_data with different amount of gestures
        for i in range(8, 21):
            _, y_train = prepare_data(train, required_gestures=i, mode=LoadMode.sequence_frame)
            _, y_test = prepare_data(test, mode=LoadMode.sequence_frame)
            print('train set gestures: {}'.format(set(y_train)))
            print('test set gestures: {}'.format(set(y_test)))
            print()


if __name__ == "__main__":
    # TODO: 测试模型
    unittest.main()
