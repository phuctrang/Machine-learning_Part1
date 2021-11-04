import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataframe = pd.read_csv('data_classification.csv', header=None)
# print(dataframe)
# X = dataframe.values[:, 0]
# Y = dataframe.values[:, 2]
# Z = dataframe.values[:, 2]
# noinspection PyUnresolvedReferences
# plt.scatter(X, Y,  marker='o')
# plt.show()
print(dataframe.values)
true_x = []
true_y = []
false_x = []
false_y = []
for item in dataframe.values:
    if item[2] ==1:
        true_x.append(item[0])
        true_y.append(item[1])
    else:
        false_y.append(item[1])
        false_x.append(item[0])
        # hinh tron: o ,,, hinh vuong: s
plt.scatter(true_x, true_y, marker='o',c='b')
plt.scatter(false_x, false_y, marker='s',c='r')
# plt.show()

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))
def border(p):
    if p >= 0.5:
        return 1
    else:
        return 0
#dự đoán
def predict(features, weight):
    z = np.dot(features, weight)
    return sigmoid(z)
"""
    z = W0 + W1xstudies(gio hoc) + W2xSleept(gio ngu)
    3x1 + 4x6 + 6x8 = 75
    [3, 4, 6]
    [1, 6, 8]   
"""


def cost_function(feature, labels, weight):
    """
    :param feature: 100x3
    :param label: 100x1 (0.1)
    :param weight:3x1 (W0 + W1xstudies(gio hoc) + W2xSleept(gio ngu))
    :return: cost
    """
    n = len(labels)
    prediction = predict(feature, weight)
    """
      prediction
      [0.4 0.5 0.6 0.7]
       """
    # y =1
    cost_class1 = -labels*np.log(prediction)
    cost_class2 = -(1-labels)*np.log(1-prediction)
    cost = cost_class2 + cost_class1
    tb = cost.sum()/n
    return tb

def update_weight(feature, labels, weight, learn_rate):
    """
    :param feature:100x3
    :param labels:100x1
    :param weight: 3x1
    :param learn_rate: float
    :return: new weight (float)
    """
    n = len(labels)
    prediction = predict(feature, weight)
    gradien = np.dot(feature.T, (prediction - labels))
    gradien = gradien/n * learn_rate
    weight = weight-gradien
    return weight
def train(feature, labels, weight, learn_rate, solanlap):
    cost_his = []
    for i in range(solanlap):
        weight = update_weight(feature, labels, weight, learn_rate)
        cost = cost_function(feature, labels, weight)
        cost_his.append(cost)
    return weight, cost_his

weight, cost = train([1, 2, 2].T, [0], 3, 1, 30 )
print(weight)
print(cost)








