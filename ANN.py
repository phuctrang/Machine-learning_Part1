import numpy as np
X = np.array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
y = np.array([0,1,1,0]).T
np.random.seed(1)
w = np.random.random((3,1)) # ma tran w: w1w2w3
# huan luyen 1000 lan: huan luyen cang nhieu lan gia tri cang chinh xac
for i in range(1000):
    z = np.dot(X, w) #x1w1+x2w2+x3w3
    output = 1/(1+np.exp(-z)) # ham sigmoid
    #update w:  x_i*(output-y)*output*(1-output)
    w = w - np.dot(X.T, (output-y)*output*(1-output))
print(1 / (1 + np.exp(-(np.dot(np.array([1,0,1]), w)))))
print(1 / (1 + np.exp(-(np.dot(np.array([0,1,1]), w)))))
print(1 / (1 + np.exp(-(np.dot(np.array([0,0,1]), w)))))
print(1 / (1 + np.exp(-(np.dot(np.array([1,1,1]), w)))))
# f(Z) ~~ Y
print(w)

########### neu x = [2]: data train ko ddc. ra ket qua toi te!










