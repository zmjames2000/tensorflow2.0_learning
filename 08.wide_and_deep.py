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


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
x_train_all, x_test, y_train_all, y_test = train_test_split(
    housing.data, housing.target, random_state=7, test_size=0.25)
x_train, x_vaild, y_train, y_vaild = train_test_split(
    x_train_all, y_train_all, random_state=7, test_size=0.25)

scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_vaild_scaler = scaler.transform(x_vaild)
x_test_scaler = scaler.transform(x_test)

'''
# 函数API
input = keras.layers.Input(shape=x_train_scaler.shape[1:])
hidden1 = keras.layers.Dense(30, activation='relu')(input)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
concat  = keras.layers.concatenate([input, hidden2])
output  = keras.layers.Dense(1)(concat)

#固化模型
model = keras.models.Model(inputs=[input], outputs=[output])
'''

# 子类API
class WideDeepModel(keras.models.Model):
    def __init__(self):
        super(WideDeepModel, self).__init__()
        '定义模型层次'
        self.hidden1_layer = keras.layers.Dense(30, activation='relu')
        self.hidden2_layer = keras.layers.Dense(30, activation='relu')
        self.output_layer  = keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        '完成模型的正向计算'
        hidden1 = self.hidden1_layer(inputs)
        hidden2 = self.hidden2_layer(hidden1)
        concat  = keras.layers.concatenate([inputs, hidden2])
        return self.output_layer(concat)

model = WideDeepModel()
model.build(input_shape=(None, 8))

model.compile(optimizer='adam',
              loss=keras.losses.mean_absolute_error)

callbacks = [keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)]

history = model.fit(x_train_scaler, y_train,
                    epochs=100,
                    validation_data=(x_vaild_scaler, y_vaild),
                    callbacks=callbacks)

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()

plot_learning_curves(history)

print('model.evaluate==================')
model.evaluate(x_test_scaler, y_test)
