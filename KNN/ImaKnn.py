import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import neighbors
import math

"""
[2.4000000000000004, 11.500000000000004, 10.199999999999989, 2]
2193.2733507773955
"""
M = 37   # 数据个数
N = 3    # 属性个数

X = np.zeros((M, N))  # 数据集
Y = np.zeros((M, 1))  # 结果（长江航运综合指数）


def load_data():  # 加载数据
    with open("afterData/new_oil_price.txt", 'r') as f:  # 加载油价
        lines = f.readlines()
        # print(len(lines))
        for i in range(1, len(lines)):
            X[i-1][0] = float(lines[i].replace("\n", ""))

    with open("afterData/sandstone_price.txt", 'r') as f:   # 加载砂石运价
        lines = f.readlines()
        # print(len(lines))
        for i in range(1, len(lines)):
            X[i-1][1] = float(lines[i].replace("\n", ""))

    with open("afterData/sea_price.txt", 'r') as f:   # 加载沿海运价综合指数
        lines = f.readlines()
        # print(len(lines))
        for i in range(1, len(lines)):
            X[i-1][2] = float(lines[i].replace("\n", ""))

    with open("afterData/composite_price.txt", 'r') as f:   # 加载长江综合航运运价指数
        lines = f.readlines()
        # print(len(lines))
        for i in range(1, len(lines)):
            Y[i-1][0] = float(lines[i].replace("\n", ""))


def make_test(x_data, y_data):
    x_test = np.zeros((M, N))
    y_test = np.zeros((M, 1))
    index = [i for i in range(M)]
    for i in range(M):
        m = random.choice(index)
        x_test[i] = x_data[m]
        y_test[i] = y_data[m]
    return x_test, y_test


def divide_data(x_data, y_data, n):  # 把数据划分成训练集和验证集 使用留出法 分层抽样  季节分层
    num = np.shape(x_data)[0]//12
    num_test = int(n*np.shape(x_data)[0])  # 必须是四的整数倍
    x_test = np.zeros((num_test, N))
    y_test = np.zeros((num_test, 1))
    x_train = np.zeros((M-num_test, N))
    y_train = np.zeros((M-num_test, 1))
    index = [m for m in range(M)]
    count = 0
    for i in range(4):
        for j in range(num_test//4):
            while True:
                offset = random.randint(0, num-1)*12+i*3+random.randint(1, 3)
                if offset in index:
                    break
            x_test[count] = x_data[offset]
            y_test[count] = y_data[offset]
            index.remove(offset)
            count += 1
    for i, j in enumerate(index):
        x_train[i] = x_data[j]
        y_train[i] = y_data[j]
    return x_test, y_test, x_train, y_train


def rescale(x_data, y_data, scale):  # 把数据缩放  返回数据集 scale为缩放因子的列表
    nx_data = x_data.copy()
    for i in range(np.shape(x_data)[0]):
        nx_data[i] = [x_data[i][n]*scale[n] for n in range(np.shape(x_data)[1])]
    return nx_data, y_data


def main():
    load_data()
    # scale = [2.4, 11.5, 10.2]
    scale = [10, 5, 5]
    x_data, y_data = rescale(X, Y, scale)
    # x_test, y_test = make_test(x_data, y_data)
    x_test, y_test, x_train, y_train = divide_data(x_data, y_data, n=0.22)
    knn = neighbors.KNeighborsRegressor(n_neighbors=2, weights="distance")
    y = knn.fit(x_train, y_train).predict(x_train)
    x = np.zeros((29, 1))
    for i in range(29):
        x[i] = i

    print('#'*10)
    plt.subplot(1, 1, 1)
    plt.plot(x, y_train, '.-', c='b', label='data')
    plt.scatter(x, y, c='g', label='predict')
    plt.axis('tight')
    plt.legend()
    plt.title("KNN")
    plt.xlabel("num")
    plt.ylabel("y_composite")
    # plt.subplot(2, 1, 2)
    # plt.plot(x, y, '.-', c='g', label='predict')
    # plt.axis('tight')
    # plt.legend()
    # plt.title("KNN")
    # plt.xlabel("num")
    # plt.ylabel("y_predict")
    plt.show()


if __name__ == '__main__':
    main()