import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import neighbors
import math

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


def rescale(x_data, y_data, scale):  # 把数据缩放  返回数据集 scale为缩放因子的列表
    nx_data = x_data.copy()
    for i in range(np.shape(x_data)[0]):
        nx_data[i] = [x_data[i][n]*scale[n] for n in range(np.shape(x_data)[1])]
    return nx_data, y_data


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


def train(x_train, y_train, k, weight="uniform"):
    knn = neighbors.KNeighborsRegressor(k, weights=weight)
    model = knn.fit(x_train, y_train)
    return model


def test_model(x_train, y_train, k, x_test, y_test, weight="distance"):
    knn = neighbors.KNeighborsRegressor(k, weights=weight)
    guess = knn.fit(x_train, y_train).predict(x_test)
    error = ((guess-y_test)**2).sum()
    # print(np.shape(x_test)[0])
    return error/np.shape(x_test)[0]


def cost(x_data, y_data, k, trials=100, n=0.12):
    error = 0.0
    for i in range(trials):
        x_train, y_train, x_test, y_test = divide_data(x_data, y_data, n)
        # model = train(x_train, y_train, k)
        error += test_model(x_train, y_train, k, x_test, y_test)
    return error/trials


def annealing_optimize(domain, costf, T=10000.0, cool=0.95, step=1):
    vec = [float(random.randint(domain[i][0], domain[i][1])) for i in range(len(domain))]
    best = 999999999999
    best_vec = vec

    while T > 0.1:
        i = random.randint(0, len(domain)-1)
        dir = random.randint(-step, step)
        vecb = vec[:]
        if i == 3:
            vecb[i] += dir
        else:
            vecb[i] += dir*0.3
        if vecb[i] < domain[i][0]:
            vecb[i] = domain[i][0]
        elif vecb[i] > domain[i][1]:
            vecb[i] = domain[i][1]

        ea = costf(vec)
        eb = costf(vecb)

        if eb < ea or random.random() < math.pow(math.e, -(eb-ea)/T):
            vec = vecb

        T *= cool
        print(vec)
        a = costf(vec)
        if a < best:
            best = a
            best_vec = vec
        print(a)
    return vec, best, best_vec


def create_cost(X, Y, n=0.12):
    def costf(vec):
        X_new, Y_new = rescale(X, Y, vec[:3])
        return cost(X_new, Y_new, int(vec[3]), n=n)
    return costf


def main():
    load_data()
    # print(X)
    # print(Y)
    weight_domain = [(0, 5), (10, 20), (5, 15), (2, 5)]
    costf = create_cost(X, Y, n=0.33)
    result, best, best_vec = annealing_optimize(weight_domain, costf, T=1000000.0, cool=0.99)
    print(result)
    print(costf(result))
    print(best_vec)
    print(best)


if __name__ == '__main__':
    main()


# plt.subplot(2, 1, 1)
# plt.scatter(X_test[:, :1], X_test[:, 1:], c='y', label='data')
# plt.scatter(X_train[:, :1], X_train[:, 1:], c='g', label='data')
# plt.axis('tight')
# plt.legend()
#
# plt.tight_layout()
# plt.show()