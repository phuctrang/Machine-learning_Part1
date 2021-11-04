import pandas as pd
import numpy as np
# data tham khao : https://www.kaggle.com/uciml/pima-indians-diabetes-database
data = pd.read_csv('datalogistic.csv')
X = data.iloc[:, :8].values
Y = data.iloc[:, -1].values
# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
#Ở đây, Bộ dữ liệu được chia thành hai phần theo tỷ lệ 75:25.
# Điều đó có nghĩa là 75% dữ liệu sẽ được sử dụng để đào tạo mô hình và 25% để thử nghiệm mô hình.
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X,Y)
print(model.intercept_)
print(model.coef_)

w0=model.intercept_[0]
w1=model.coef_[0,0]
w2=model.coef_[0,1]
w3=model.coef_[0,2]
w4=model.coef_[0,3]
w5=model.coef_[0,4]
w6=model.coef_[0,5]
w7=model.coef_[0,6]
w8=model.coef_[0,7]
#

# test = [[34, 78]]
# result = model.predict(test)
# print(result)

import matplotlib.pyplot as plt
x1 = X[:, 0]
x2 = X[:, 1]
x3 = X[:, 2]
x4 = X[:, 3]
x5 = X[:, 4]
x6 = X[:, 5]
x7 = X[:, 6]
x8 = X[:, 7]
color = ['red' if value == 1 else 'blue' for value in Y]
plt.scatter(x1,Y, color=color)
plt.scatter(x2,Y, color=color)
plt.scatter(x3,Y, color=color)
plt.scatter(x4,Y, color=color)
plt.scatter(x5,Y, color=color)
plt.scatter(x6,Y, color=color)
plt.scatter(x7,Y, color=color)
plt.scatter(x8,Y, color=color)

plt.title('Logistic regression for data')

yy = 1/(1+np.exp(-(w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6 + w7*x7 + w8*x8 + w0)))
plt.plot(X, yy)
plt.show()
# 78% !!!
kq = model.score(X,Y)
print(kq)










