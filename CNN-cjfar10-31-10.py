import numpy as np
import pandas as pd
import numpy as nb
import tensorflow
import keras
# anh 10 lop
from keras.datasets import  cifar10
from keras.models import Sequential
from keras.layers import Dense , Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras import datasets, layers, models
# load data

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# 50k train
# print(x_train.shape)
# # 10k test
# print(x_test.shape)
import matplotlib.pyplot as plt
import matplotlib.image as img
# 10 class
class_name = ['May bay', 'Xe con', 'Chim', 'Meo', 'Con Huou', 'Cho', 'Nhai', 'Ngua', 'Tau thuy', 'Xe tai']
plt.figure(figsize=(10,10))

# show 25 anh
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_name[y_train[i][0]])
# anh cac doi tuong
# plt.show()
y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
y_test = tensorflow.keras.utils.to_categorical(y_test, 10)

# print(y_train.shape, y_test.shape)
#OK
# create model no ron
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# tranh qua trinh bi overfing 25%
model.add(Dropout(0.25))


model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

##############
# xong trich xuat dac trung
##############
# gian ra thanh mang 1 chieu
model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
# output
model.add(Dense(10, activation='softmax'))

# print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255
H = model.fit(x_train, y_train, validation_data=(x_test, y_test),
              batch_size=128, epochs=1, verbose=1)
# print(H)
# 76.7%
print(model.evaluate(x_test, y_test))
## do chinh xac thap
model.save('modeltrained.h1')

from skimage.transform import resize
import matplotlib.image as img
img1 = img.imread('xe_cnn.png')
plt.imshow(img1)
img2 = resize(img1, (32,32))
print(img2.shape)
# ( 32 , 32 , 3)
# chuyen kich thuoc
img3 = img2.reshape(1,32,32,3)

y_mu = model.predict(img3)
print(y_mu)
index = np.argmax(y_mu)
print('chi so lop lon nhat (ket qua)',index)
print('anh nay la ảnh của: ', class_name[index])

# in ra 10 so softmax
# lop xe con la so cao nhat. đúng kết quả !!!




# lay model da train: da save
from keras.models import load_model
model1 = load_model('modeltrained.h1')
y_mu = model1.predict(img3)
print(y_mu)
index = np.argmax(y_mu)
print('chi so lop lon nhat (ket qua)',index)
print('anh nay la ảnh của: ', class_name[index])










