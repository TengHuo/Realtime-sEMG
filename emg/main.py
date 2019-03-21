# -*- coding: UTF-8 -*-
# main.py
# @Time     : 13/Dec/2018
# @Auther   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#

from scipy.io import loadmat
import pandas as pd
import csv
from os import walk

for root, dirs, files in walk("./data/Mat_Data/"):
    print('ROOT:{}'.format(root))
    print('DIR:{}'.format(dirs))
    # for filename in files:
    #     print(filename)

# mat = loadmat('data/Mat_Data/dba-preprocessed-001/001-001-001.mat')
# data_array = mat['data']

# print(data_array)

# csv_file = open('./test.csv', 'w')
# writer = csv.writer(csv_file)
# writer.writerows(data_array)

# # write the data in a new csv file

# # print(type(mat))
# # print(mat.keys())
# # print(len(mat['data'][0]))
