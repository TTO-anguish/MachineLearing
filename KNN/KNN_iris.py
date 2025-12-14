#!/usr/bin/env python
# coding=utf-8
'''
FilePath     : /MachineLearing/KNN/KNN_iris.py
Description  :  改方案没有交叉验证，需要添加交叉验证环节，和更改超参
Author       : fei.zong zongfei2019@outlook.com
Version      : 0.0.1
LastEditors  : fei.zong zongfei2019@outlook.com
LastEditTime : 2025-12-14 22:20:02
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

def dm03_split_train_text():
    iris_data = load_iris()
    #分割训练集和测试集,8:2分割
    #返回值是个元组，参1:特征数据  标签数据   测试集比例  随机种子
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=42)
    print(f'训练集特征数据:\n{x_train}, 个数: {len(x_train)}')
    print(f'测试集特征数据:\n{x_test}, 个数: {len(x_test)}')
    print(f'训练集标签数据:\n{y_train}, 个数: {len(y_train)}')
    print(f'测试集标签数据:\n{y_test}, 个数: {len(y_test)}')

#完整案例
def dm04_iris_evaluate_test():
    #1.加载数据集
    iris_data = load_iris()
    #2.分割训练集和测试集,8:2分割(数据集预处理)
    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=42)
    #3.特征工程(提取、预处理)，因为源数据只有4个特征列，而且都用，所有不用提取
    #预处理：源数据4列特征插值不大，无需做特征处理，但是加入会让代码更完善
    #3.1 创建标准化对象
    transfer = StandardScaler()
    #3.2 标准化训练集特征数据
        #fit_transform:兼具fit和transfer功能，适用于第一次标准化时候使用
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)  #transform:只对测试集进行标准化

    #4.选择模型/模型训练
    estimator = KNeighborsClassifier(n_neighbors=3)  #n_neighbors:邻居数
    estimator.fit(x_train, y_train)  #训练模型

    #5.模型预测
    #5.1.预测测试集
    y_predict = estimator.predict(x_test)  #预测测试集
    print(f'预测测试集结果:\n{y_predict}')

    #5.2.预测训练集（对新的数据集进行测试）
    my_data = [[5.1, 3.5, 1.4, 0.2], [6.7, 3.1, 4.4, 1.4], [5.9, 3.0, 5.1, 1.8]]
    my_data = transfer.transform(my_data)  #标准化
    my_predict = estimator.predict(my_data)  #预测新的数据集
    print(f'预测新的数据集结果:\n{my_predict}')

    #5.3查看上述数据集，每种分类的预测概率
    my_predict_proba = estimator.predict_proba(my_data)  #预测概率
    print(f'预测新的数据集每种分类的概率:\n{my_predict_proba}')

    #模型评估
    #法1：直接评分，基于训练集的特征和标签
    print(f'准确率: {estimator.score(x_train, y_train)}')

    #法2：基于测试集标签和预测结果进行评估
    print(f'准确率: {accuracy_score(y_test, y_predict)}')

if __name__ == "__main__":
    #dm01_loadiris()
    #dm02_ShowIris()
    #dm03_split_train_text()
    dm04_iris_evaluate_test()


#关于交叉验证
'''
1.将训练集划分成n份（n=4或5），每次取一份作为验证集，其他作为训练集
2.训练n次，每次使用不同的验证集
交叉验证法原理：将数据集划分为 cv=4 份

第一次：把第一份数据做验证集，其他数据做训练
第二次：把第二份数据做验证集，其他数据做训练
… 以此类推，总共训练4次，评估4次
使用训练集+验证集多次评估模型，取平均值做交叉验证为模型得分
若k=5模型得分最好（即第5份最好），再使用全部训练集（训练集+验证集）对k=5模型再训练一边，再使用测试集对k=5模型做评估(k=5就是n值)

'''