import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
"""Dữ liệu bênh nhân ung thư 30 biến độc lập"""
data = load_breast_cancer()
x = data.data
y = data.target
# print(x.shape, y.shape)
model = GaussianNB()
model.fit(x,y)
GaussianNB()
ymu = model.predict(x)
# print(ymu)
x1 = x[:, 0]
x2 = x[:, 1]
color = ['red' if value == 1 else 'blue' for value in y]
plt.scatter(x1, x2, marker='o', color=color)
plt.show()
print(model.score(x,y))

# plt.scatter(x,y)
# plt.scatter([x[ymu==0,1], x[ymu==0,2]], c = 'b')
# plt.scatter([x[ymu==1,1], x[ymu==1,2]], c = 'b')
# plt.show()



