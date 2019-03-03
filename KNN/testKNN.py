import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import neighbors
import math

X = np.zeros((37, 2))
Y = np.zeros((37, 1))

with open("afterData/new_oil_price.txt", 'r') as f:
    lines = f.readlines()
    print(len(lines))
    for i in range(1, len(lines)):
        X[i-1][0] = float(lines[i].replace("\n", ""))

with open("afterData/sandstone_price.txt", 'r') as f:
    lines = f.readlines()
    print(len(lines))
    for i in range(1, len(lines)):
        X[i-1][1] = float(lines[i].replace("\n", ""))

with open("afterData/composite_price.txt", 'r') as f:
    lines = f.readlines()
    print(len(lines))
    for i in range(1, len(lines)-1):
        Y[i-1][0] = float(lines[i].replace("\n", ""))

print(X)
X_test = np.zeros((2, 2))
X_train = np.zeros((35, 2))
Y_test = np.zeros((2, 1))
Y_train = np.zeros((35, 1))

index = [i for i in range(37)]

result = 0
Y_result = np.zeros((2, 1))
for x in range(10):
    index = [i for i in range(37)]
    for i in range(2):
        num = random.choice(index)
        index.remove(num)
        X_test[i] = X[num]
        Y_test[i] = Y[num]

    for i, j in enumerate(index):
        X_train[i] = X[j]
        Y_train[i] = Y[j]

    n_neighbors = 3

    knn = neighbors.KNeighborsRegressor(n_neighbors, weights="uniform")
    pre = knn.fit(X_train, Y_train)
    # Z_test = pre.predict(X_test)
    new = pre.score(X_test, Y_test)
    pre_data = pre.predict(X_test)
    print(pre_data)
    print("@@@@@@@@@@@@@@@@@@@@@")
    print(Y_test)
    # Y_result += pre_data
    # Y_error += pre_data-Y_test
    print(new)
    result += new

print(result/10)
# print(Y_result/10)
# print(Y_error/10)


# fig = plt.figure()
# ax = fig.add_subplot(211, projection="3d")
# ax.plot_surface(X_test[:, :1], X_test[:, 1:], Z_test, rstride=1, cstride=1, cmap='rainbow', label='prediction')
# al = fig.add_subplot(212, projection="3d")
# al.plot_surface(X_test[:, :1], X_test[:, 1:], Y_test, rstride=1, cstride=1, cmap='rainbow', label='data')
# fig.show()

plt.subplot(2, 1, 1)
plt.scatter(X_test[:, :1], X_test[:, 1:], c='y', label='data')
plt.scatter(X_train[:, :1], X_train[:, 1:], c='g', label='data')
plt.axis('tight')
plt.legend()

plt.tight_layout()
plt.show()