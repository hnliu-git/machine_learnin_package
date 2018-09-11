#coding=utf-8

"""
@author: Teren
合并两个同类型矩阵
"""

import numpy as np
import pickle
from scipy import sparse
import time

t_start = time.time()

fea1 = ""
fea2 = ""
isSparse = True
output = ""

"""
读取特征
"""

f1 = open(fea1, 'rb')
x_train_1, y_train, x_test_1 = pickle.load(f1)
f1.close()

f2 = open(fea2, 'rb')
x_train_2, y_train,x_test_2 = pickle.load(f2)
f2.close()

"""
融合保存
"""
if isSparse:
    x_train = sparse.hstack([x_train_1, x_train_2])
    x_test = sparse.hstack([x_test_2, x_test_1])
else:
    x_train = np.concatenate([x_train_1, x_train_2], axis=1)
    x_test = np.concatenate((x_test_1, x_test_2), axis=1)

data = (x_train, y_train, x_test)
fp = open(output, 'wb')
pickle.dump(data, fp)
fp.close()

t_end = time.time()
print("已融合，共耗时：{}min".format((t_end-t_start)/60))
