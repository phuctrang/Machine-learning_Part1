from sklearn.datasets import load_digits
data = load_digits()
# x = 4cot
X = data.data
Y = data.target
print(X.shape)
print(Y)
# print(X)
# print(Y)
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# model.fit(X,Y)
# # print(model.intercept_)
# # print(model.coef_)
# pre = model.predict(X)
# diem = model.score(X,Y)
# print(pre)
# print(diem)
# ymu = model.predict(X)
# import matplotlib.pyplot as plt
# plt.scatter(X[ymu==0,2], X[ymu==0,3], c='b')
# plt.scatter(X[ymu==1,2], X[ymu==1,3], c='r')
# plt.scatter(X[ymu==2,2], X[ymu==2,3], c='g')
# plt.show()



