# XOR

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, AveragePooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import tensorflow as tf
import logging
import numpy as np

model_1 = Sequential()
model_1.add(Dense(8, input_dim=2, activation='relu'))
model_1.add(Dense(1, activation='sigmoid'))
model_1.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))

# print(model_1.summary())

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
model_1.fit(X, y, batch_size=1, epochs=1000, verbose=0)

print("Проверка обученной сети: ")
print("XOR(0, 0):", model_1.predict(np.array([[0, 0]]), batch_size=1, verbose=0))
print("XOR(0, 1):", model_1.predict(np.array([[0, 1]]), batch_size=1, verbose=0))
print("XOR(1, 0):", model_1.predict(np.array([[1, 0]]), batch_size=1, verbose=0))
print("XOR(1, 1):", model_1.predict(np.array([[1, 1]]), batch_size=1, verbose=0))

# Параметры первого уровня
W1 = model_1.get_weights()[0]
b1 = model_1.get_weights()[1]

# Параметры второго уровня
W2 = model_1.get_weights()[2]
b2 = model_1.get_weights()[3]

print("W1:", W1)
print("b1:", b1)
print("W2:", W2)
print("b2:", b2)