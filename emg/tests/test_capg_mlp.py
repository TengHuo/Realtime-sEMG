# -*- coding: UTF-8 -*-
# test_capg_mlp.py
# @Time     : 28/Mar/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
# TODO: add more test case

import os
import unittest
from emg.classification import CapgMLP

class TestCapgMLP(unittest.TestCase):

    def setUp(self):
        # TODO: create a dataset for test
        self.test_data_path = ''
        self.data = None
        return super().setUp()

    def tearDown(self):
        # TODO: delete test dataset
        os.remove(self.test_data_path)
        return super().tearDown()

    def test_model_init(self):
        pass



if __name__ == "__main__":
    # TODO: 测试模型
    unittest.main()
