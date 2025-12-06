
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def dataset(): 
    iris = load_iris()
    #print("iris.data.shape: \n", iris)
    # print("查看数据集描述：\n", iris.DESCR)
    print("查看数据集描述：\n", iris["DESCR"])
    print("查看数据集特征名称：\n", iris.feature_names)

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("x_train.shape: ", x_train.shape)
    print("x_test.shape: ", x_test.shape)
    return None

#字典特征提取
def dict_demo():
    data = [{"name": "zhangsan", "age": 18}, {"name": "lisi", "age": 20}, {"name": "wangwu", "age": 19}]
    from sklearn.feature_extraction import DictVectorizer
    #实例化一个DictVectorizer对象
    transfer = DictVectorizer(sparse=False)  #sparse=False表示输出为数组格式  默认为true，输出为稀疏矩阵格式,稀疏矩阵节省内存,也可以使用停用词stop_words参数来设置停用词
    #调用fit_transform方法
    data_new = transfer.fit_transform(data)
    print("data_new: \n", data_new)
    print("特征名称：\n", transfer.get_feature_names_out())

#文本特征提取，只能提取英文文本，中文可采用结巴（jieba）分词等方式处理
def cont_demo():
    data = ["life is short", "I like python", "I love machine learning"]
    from sklearn.feature_extraction.text import CountVectorizer  #统计每个样本特征词出现的个数
    #实例化一个CountVectorizer对象
    transfer = CountVectorizer()
    #调用fit_transform方法
    data_new = transfer.fit_transform(data)
    print("data_new: \n", data_new.toarray())


if __name__ == "__main__":
    #dataset()
    cont_demo()