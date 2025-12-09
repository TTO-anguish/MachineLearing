#!/usr/bin/env python
# coding=utf-8
'''
FilePath     : /MachineLearing/KNN/KNN_iris.py
Description  :  
Author       : fei.zong zongfei2019@outlook.com
Version      : 0.0.1
LastEditors  : fei.zong zongfei2019@outlook.com
LastEditTime : 2025-12-10 01:07:50
Copyright    : G AUTOMOBILE RESEARCH INSTITUTE CO.,LTD Copyright (c) 2025.
'''
from sklearn.datasets import load_iris         #加载数据库
from sklearn.model_selection import train_test_split     #分割训练集和测试集
from sklearn.preprocessing import StandardScaler     #标准化
from sklearn.neighbors import KNeighborsClassifier     #KNN分类器
from sklearn.metrics import accuracy_score     #准确率
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import savemat  #保存.mat文件,用于Matlab画图
import os

def dm01_loadiris():
    #加载鸢尾花数据集
    iris_data = load_iris()
    #查看数据集
    #print(f'数据集"{iris_data}')
    #print(f'数据集类型"{type(iris_data)}')
    print(f'数据集所有的键:{iris_data.keys()}') #数据集所有的键:dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
    print(f'标签对应的名称:{iris_data.target_names}') #标签对应的名称:['setosa' 'versicolor' 'virginica']
    print(f'特征名称:{iris_data.feature_names}') #特征名称:['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    print(f'数据集描述:{iris_data.DESCR}') #数据集描述:.. _iris_dataset:

def dm02_ShowIris():
    iris_data = load_iris()

    #把鸢尾花数据集封装成 DataFrame对象
    iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    iris_df['target'] = iris_data.target  #添加标签列
    #print(iris_df.head())  #查看前5行数据
    print(iris_df) 

    #可视化数据集
    sns.pairplot(iris_df, x_vars='sepal length (cm)', y_vars='sepal width (cm)', hue='target') #hue是根据什么分组

    file_path = os.path.abspath('iris_data.mat')
    print(f"iris_data.mat 文件保存在：{file_path}")
    savemat('iris_data.mat', {'iris_data': iris_data.data, 'target': iris_data.target})

    #设置图表标题和显示图表
    plt.title('Iris Dataset')
    plt.show()

if __name__ == "__main__":
    #dm01_loadiris()
    dm02_ShowIris()