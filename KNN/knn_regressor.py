#!/usr/bin/env python
# coding=utf-8
'''
FilePath     : /MachineLearing/KNN/knn_regressor.py
Description  :  None
Author       : fei.zong zongfei2019@outlook.com
Version      : 0.0.1
LastEditors  : fei.zong zongfei2019@outlook.com
LastEditTime : 2025-12-08 00:26:36
Copyright    : G AUTOMOBILE RESEARCH INSTITUTE CO.,LTD Copyright (c) 2025.
'''

from sklearn.neighbors import KNeighborsRegressor


'''
差值:(3,9,11)   (2,10,10)  (0,1,0)  (1,0,2)
距离:14          204         1         2.24
预测值：(0.2 + 0.3 + 0.4) / 3 = 0.3
'''
x_train = [[0, 1, 1], [1, 1, 0], [3, 10, 10], [4, 11, 12]]
y_train = [0.1, 0.2, 0.3, 0.4]
x_test = [[3, 11, 10]]

#创建模型
estimator = KNeighborsRegressor(n_neighbors=3)
estimator.fit(x_train, y_train)

y_pre = estimator.predict(x_test)

print(f'预测值为：{y_pre}')