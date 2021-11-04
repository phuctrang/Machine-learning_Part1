import  matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
data = pd.read_csv('cao_nang.csv')
X = data[['cao']]
Y = data['nang']

plt.xlabel("Chieu cao")
plt.ylabel("Ket qua can nang")

plt.scatter(X,Y,color='red')
model = linear_model.LinearRegression()
#train
model.fit(X,Y)
plt.plot(X, model.predict(X), color='blue')
# mang 2 chieu can dinh dang lai du lieu: X.reshape(-1,1)
plt.show()
print("he so ngau nhien")
print(model.coef_)
print("gia tri khoi dau cua y khi X=0")
print(model.intercept_)

print("chieu cao du doan khi can nang la 172kg: ")
print(model.predict([[172]]))





