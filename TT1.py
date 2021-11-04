import pandas as pd
import matplotlib.pyplot as plt
dataframe = pd.read_csv('Advertising.csv')
print(dataframe)
X = dataframe.values[0:, 2]
Y = dataframe.values[0:, 4]
# # print(X) radio (attibute)
# print(Y) sales (label)
# plt.scatter(X, Y, marker='o')
# plt.show()
# Hàm dự đoán :
#sale = weight * radio + bias  (Tạo một dự đoán )

def predict(new_radio, weight, bias):
    return  weight*new_radio + bias
# Hàm chi phí
def cost_function(X, Y, weight, bias):
    # mx + b: gia tri dự đoán
    # y: Giá trị thực
    # m = weight , b = bias
    n = len(X)
    sum_error = 0
    for i in range(n):
        # binh phuong
        sum_error += (Y[i]-weight*X[i]+bias)**2
    return sum_error/n
## Complete video 1.
def update_weight(X, Y, weight, bias, learn_rate):
    # learn_rate: Tốc độ học  .
    n = len(X)
    weight_temp = 0.0
    bias_temp = 0.0
    for i in range(n):
        # đạo hàm theo m (Weight)
        weight_temp += -2*X[i]*(Y[i]-((weight*X[i]) + bias))
        # đạo hàm theo b (Bias)
        bias_temp += -2*(Y[i]-((weight*X[i]) + bias))
        # tính trung bình
        weight -= (weight_temp/n) * learn_rate
        bias -= (bias_temp/n) * learn_rate
    return weight, bias
# hamf training
def training(X, Y, weight, bias, learn_rate, solanlap):
    cost_history = []
    for i in range(solanlap):
        weight,bias = update_weight(X, Y, weight, bias, learn_rate)
        cost=cost_function(X, Y, weight, bias)
        cost_history.append(cost)
    return  weight, bias, cost_history

weight,bias, cost = training(X, Y, 0.03, 0.0014, 0.001, 200)
print('result: ')
print(weight)
print(bias)
print(cost)
dudoan = predict(19, weight, bias)
print('ket qua du doan: ')
print(dudoan)
solanlap = [i for i in range(200)]
plt.plot(solanlap, cost)
plt.show()















