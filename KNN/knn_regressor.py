'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-12-07 00:59:25
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-12-07 01:14:49
FilePath: \Machine Learing\KNN\knn_regressor.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from sklearn.neighbors import KNeighborsRegressor


'''
差值:(3,9,11)   (2,10,10)  (0,1,0)  (1,0,2)
距离:14          204         1         2.24
预测值：(0.2 + 0.3 + 0.4) / 3 = 0.3
'''
x_train = [[0, 1, 1], [1, 1, 0], [3, 10, 10], [4, 11, 12]]
y_train = [0.1, 0.2, 0.3, 0.4]
x_test = [3, 11, 10]

#创建模型
estimator = KNeighborsRegressor(n_neighbors=3)
estimator.fit(x_train, y_train)

y_pre = estimator.predict(x_test)

print(f'预测值为：{y_pre}')