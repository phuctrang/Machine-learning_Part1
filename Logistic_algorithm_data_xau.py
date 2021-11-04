import pandas as pd
import numpy as np
data = pd.read_csv('ex2data2_xau.csv')
X = data.iloc[:, :2].values
Y = data.iloc[:, -1].values
print(X)
print(Y)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X,Y)
print(model.intercept_)
print(model.coef_)

w0=model.intercept_[0]
w1=model.coef_[0,0]
w2=model.coef_[0,1]
#

test = [[34, 78]]
result = model.predict(test)

print(w0, w1, w2)
print(result)

import matplotlib.pyplot as plt
x1 = X[:, 0]
x2 = X[:, 1]
color = ['red' if value == 1 else 'blue' for value in Y]
plt.scatter(x1, x2, marker='o', color=color)
plt.xlabel('X1 input feature')
plt.ylabel('X2 input feature')
plt.title('Perceptron regression for X1, X2')

# print("x1 = ", x1)
#
yy = (-(w0 + w1*x1)) / w2
plt.plot(x1, yy)
plt.show()

# giong voi cach cua thay!
# ymin, ymax = plt.ylim()
# ##     w la:                     w1 va w2
# w = model.coef_[0]
#
# a = -w[0] / w[1]
# xx = np.linspace(ymin, ymax)
# yy = a * xx - (model.intercept_[0]) / w[1]
#
# # Plot the hyperplane
# plt.plot(xx, yy, 'k-')
#
# model.predict(X)
# plt.plot(X)
ok = model.score(X,Y)
print("% model dung: ",ok )
# khac
y1 = model.predict(X)
y = np.array(Y)
dem = 0
for i in range(len(y)):
    if y[i] == y1[i]:
        dem = dem +1
print(dem)









