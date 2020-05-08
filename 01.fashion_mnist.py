import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import sklearn
import os
import sys
import time
import tensorflow as tf
import pprint

from tensorflow import keras

print('Tensorflows Version:{}'.format(tf.__version__))
# print('Is gpu available:{}'.format(tf.test.is_gpu_available()))
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 将数据分为训练, 验证, 测试
x_train, y_train, x_vaild, y_vaild = train_test_split(
    x_train_all, y_train_all, random_state=7, test_size=0.25)#param: test_size=0.25
# 归一化处理  x = (x -平均值）/std 做值的归一化处理
scaler = StandardScaler()
# [none, 28, 28] ---> [none, 784] ----> [none, 28, 28]
x_train_scaler = scaler.fit_transform(x_train.astype(np.float33).reshape(-1,1)).reshape(-1, 28, 28)
x_vaild_scaler = scaler.transform(x_vaild.astype(np.float32).reshape(-1,1)).reshape(-1, 28, 28)
x_test_scaler  = scaler.transform(x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])

model.compile(optimizer='adam',
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['acc'])
model.summary()
history = model.fit(x_train_scaler, y_train,
                    epochs=100,
                    validation_data=(x_vaild_scaler, y_vaild))

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()

plot_learning_curves(history)

model.evaluate(x_test_scaler, y_test)
