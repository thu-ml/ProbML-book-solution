'''
avila数据集:https://www.kaggle.com/datasets/hritaban02/avila-dataset?resource=download

iris数据集本身规模较小，线性核与高斯核性能差距不大。

Avila数据集上，高斯核对于复杂数据分布的分类性能显著高于线性核。

除了kernel类型以外，SVM常用参数还包括正则化参数C，对模型参数起到正则化的作用，C越大，正则化越弱，
在Avila上可以直观观察到分类性能会进一步提升。

其他参数可根据sklearn的svm参数列表自行实验。
'''

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

def load_iris_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    print(X.shape, y.shape)
    return X, y

def load_avila_data():
    df = pd.read_csv('ch07/avila_combined.txt', delimiter=",", header=None)
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    print(X.shape, y.shape)
    return X, y

if __name__ == '__main__':

    print("====Iris====\n\n")
    X, y = load_iris_data()
    # X, y = load_avila_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))

    print("\n\n====Avila====\n\n")

    X, y = load_avila_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    clf = svm.SVC(kernel='rbf', C=30)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))
