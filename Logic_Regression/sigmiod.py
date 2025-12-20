#!/usr/bin/env python
# coding=utf-8
'''
FilePath     : /AI_python/MachineLearing/Logic_Regression/sigmiod.py
Description  :  癌症预测，逻辑回归
Author       : fei.zong zongfei2019@outlook.com
Version      : 0.0.1
LastEditors  : fei.zong zongfei2019@outlook.com
LastEditTime : 2025-12-21 03:08:01
Copyright    : G AUTOMOBILE RESEARCH INSTITUTE CO.,LTD Copyright (c) 2025.
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer # Load the breast cancer dataset
import matplotlib.pyplot as plt
# Load the dataset

data = pd.read_csv('./data/breast-cancer-wisconsin.csv') #目前使用自己自带的,这个文件的空值用问号表示的，要去掉
data.info()  # 查看数据集信息(不需要print)

# Preprocess the dataset
data.replace('?', np.nan, inplace=True)  # 替换问号为NaN，Not a Number” 的缩写，表示缺失值或无效数字

#缺失值处理->删除
data.dropna(axis=0, inplace=True)  # 删除包含NaN的行,axis=0表示行
#data.info() #查看处理后的数据

#3，特征工程
x = data.iloc[:, 1:-1]  #按照行号，列索引所有数据，:表示所有行，1:-1表示从第一列到最后一列，[ )，左开右闭
#y = data.iloc[:, -1]  #获取最后一列
#y = data.iloc['Class'] #获取最后一列
y = data.Class #获取最后一列,最后一个没中文，没空格，什么都没有，可以直接.出来

print(x[:5]) #打印特征数据 x 的前5行，方便快速查看数据内容。
print(y[:5])
print(x.shape)
print(y.shape)

#3.切割数据集和测试集
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=22)

#3.4创建标准化对象
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

#4 模型训练
estimator = LogisticRegression()   #这里可以不用传参数

estimator.fit(x_train, y_train)

#预测
y_pre = estimator.predict(x_test)
print(f'预测值为：{y_pre}')

#评估
print(f'预测评估，准确率{estimator.score(x_test, y_test)}')
print(f'预测评估，准确率{accuracy_score(y_test, y_pre)}')

# 思考：逻辑回归模型能用准确率来评测吗？
# 答案：可以，但是结果不精准，因为逻辑回归模型主要用于二分类，即：A类还是B类，不能说97%的A类，3%的B类。
# 所以要通过混淆矩阵来评测，即：准确率，召回率，F1值(F1-Score)，ROC曲线，AUC值。