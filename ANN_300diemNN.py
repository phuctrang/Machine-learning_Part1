import numpy as np
import matplotlib.pyplot as plt
N = 100 # number of points per class\n"
d0 = 2 # dimensionality\n",
C = 3 # number of classes\n",
X = np.zeros((d0, N*C)) # data matrix (each row = single example)
y = np.zeros(N*C, dtype='uint8') # class labels"
for j in range(C):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)  # radius
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
    X[:, ix] = np.c_[r * np.sin(t), r * np.cos(t)].T
    y[ix] = j
plt.plot(X[0, :N], X[1, :N], 'bs', markersize = 7)
plt.plot(X[0, N:2*N], X[1, N:2*N], 'ro', markersize = 7)
plt.plot(X[0, 2*N:], X[1, 2*N:], 'g^', markersize = 7)
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])
plt.savefig('EX.png', bbox_inches='tight', dpi = 600)
#plt.show() # AVATA :)))

from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
model = Sequential()
model.add(Dense(50, activation='relu',input_shape=(2,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(3,activation='softmax'))
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    )
"""""""""""""""not .........."""""""""""""""



