#!/usr/bin/env python
# coding=utf-8
'''
FilePath     : /MachineLearing/KNN/KNN_GridSearchCV.py
Description  :  交叉验证
Author       : fei.zong zongfei2019@outlook.com
Version      : 0.0.1
LastEditors  : fei.zong zongfei2019@outlook.com
LastEditTime : 2025-12-18 01:13:41
Copyright    : G AUTOMOBILE RESEARCH INSTITUTE CO.,LTD Copyright (c) 2025.
'''

#交叉验证和网格搜素

'''
交叉验证解释：

原理：

把数据分成n份，例如分成：4份 → 也叫：4折交叉验证。

第1次：把第1份数据作为 验证集（测试集），其它作为训练集，训练模型，模型预测，获取：准确率 → 准确率1
第2次：把第2份数据作为 验证集（测试集），其它作为训练集，训练模型，模型预测，获取：准确率 → 准确率2
第3次：把第3份数据作为 验证集（测试集），其它作为训练集，训练模型，模型预测，获取：准确率 → 准确率3
第4次：把第4份数据作为 验证集（测试集），其它作为训练集，训练模型，模型预测，获取：准确率 → 准确率4
然后计算上述的 4次准确率的 平均值，作为：模型最终的 准确率。

假设第4次最好（准确率最高），则：用全部数据（训练集 + 测试集）训练模型，再次用（第4次的）测试集对模型测试。

目的：为了让模型的最终验证结果更准确。


网格搜索：

目的/作用：寻找最优超参数。

原理：
接收超参可能出现的值，然后针对于 超参的每个值进行 交叉验证，获取到 最优超参组合。

超参数：需要用户手动录入的数据，不同的超参（组合），可能会影响模型的最终评测结果。

大白话解释：

网格搜索 + 交叉验证，本质上指的是 GridSearchCV 这个API，它会帮我们寻找最优超参（供参考）。

'''

from sklearn.datasets import load_iris         #加载数据库
from sklearn.model_selection import train_test_split, GridSearchCV     #分割训练集和测试集
from sklearn.preprocessing import StandardScaler     #标准化
from sklearn.neighbors import KNeighborsClassifier     #KNN分类器
from sklearn.metrics import accuracy_score     #准确率
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import savemat  #保存.mat文件,用于Matlab画图
import os

iris_data = load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.3, random_state=22)

#创建标准化对象
transfer = StandardScaler()

x_train = transfer.fit_transform(x_train)  #fit_transform()方法：先拟合数据，再转换数据
x_test = transfer.transform(x_test)  #transform()方法：只转换数据

#KNN
estimator = KNeighborsClassifier()

#1.定义字典
param_dict = {'n_neighbors':[i for i in range(1, 11)]}

#2.创建GridSearch对象 -> 寻找最优超参数，使用网格搜素+ 交叉验证
estimator = GridSearchCV(estimator, param_dict, cv=4)
'''
内部主要做了以下几件事：

参数组合遍历
把 param_dict 里所有参数组合（如 n_neighbors=1,2,...,10）都列出来。

交叉验证
对每一组参数组合，做 4 折交叉验证（cv=4）：

把训练集分成4份，每次用1份做验证集，剩下3份做训练集。
用当前参数组合训练模型，再在验证集上评估准确率。
4次轮流，每份都做一次验证集，得到4个准确率。
计算平均分数
对每组参数，取4次交叉验证的平均准确率，作为该参数组合的得分。

选出最优参数
比较所有参数组合的平均分数，选出分数最高的那组参数，作为“最优超参数”。

自动重训练
用最优参数，在全部训练集上重新训练模型，得到最终模型。

结果保存
你可以通过 .best_params_ 查看最优参数，通过 .best_score_ 查看最优分数，通过 .best_estimator_ 拿到最优模型。



此时的 estimator（GridSearchCV对象）确实还没有“数据”，只是把模型、参数范围、交叉验证方式等配置好了。
真正的数据训练和交叉验证，是在你后面执行：

'''

#训练
estimator.fit(x_train, y_train)
print(f'最优评分:{estimator.best_score_}')
print(f'最优超参组合{estimator.best_params_}')
print(f'最优估计器对象{estimator.best_estimator_}')
print(f'具体的交叉验证结果{estimator.cv_results_}')

'''
cv、test_size、random_state:这些都会影响最终的结果。
cv:交叉验证的折数，越大越精确，但计算量也越大。
test_size:测试集的比例，通常是0.2或0.3。

'''

estimator = KNeighborsClassifier(n_neighbors=estimator.best_params_['n_neighbors'])
#重新训练模型
estimator.fit(x_train, y_train)
#预测测试集
y_predict = estimator.predict(x_test)
#评估模型
print(f'测试集准确率: {accuracy_score(y_test, y_predict)}')