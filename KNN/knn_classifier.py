from sklearn.neighbors import KNeighborsClassifier


#准备测试集和训练集
x_train = [[0], [1], [2], [3]]   #定义成列向量，训练集的特征数据
y_train = [0, 0, 1, 1]           #训练集的标签数据
x_test = [[5]]

#创建KNN分类模型
estimater = KNeighborsClassifier(n_neighbors=2)

#模型训练
estimater.fit(x_train, y_train)

#模型预测
y_pre = estimater.predict(x_test)

#打印预测结果
print(f'预测值为：{y_pre}')