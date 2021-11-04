import math

import pandas as pd
import numpy as np
data = pd.read_csv('databai1.csv')
X = data.iloc[:, :1].values
Y = data.iloc[:, -1].values
print(X)
print(Y)
color = ['red' if value == 1 else 'blue' for value in Y]
import matplotlib.pyplot as plt
plt.scatter(X,Y, color = color)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X,Y)

# ax + b             a la w1, b la w0
w0 = model.intercept_[0]
w1 = model.coef_[0,0]
print(w1, w0)

# y = 1/1+e^ax+b
yy = 1/(1+np.exp(-w1*X - w0))
plt.plot(X,yy)


kq =model.predict([[2.5]])
y_mu = model.predict(X)

print(kq)
print(y_mu)
# plt.plot(y_mu)
print(model.score(X,Y))
# plt.show()