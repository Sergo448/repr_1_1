# Первая созданная и обученная сверточная нейронная сеть
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense


np.random.seed(123)
# Объявили все необходимые библиотеки

# Подгрузиди БД с изображениями
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Вывели информацию о БД
print("Тренировочный набор данных", X_train.shape)
print("Метки тренировочного набора данных", y_train.shape)
print("Тестовый набор данных", X_test.shape)
print("Метки тестового набора данных", y_train.shape)

"""
plt.imshow(X_train[1])
plt.show()

# Вывели изображение из тренировочного надора данных с цветовой шкалой

plt.figure()
plt.imshow(X_train[1])
plt.colorbar()
plt.grid(False)
plt.show()

# Вывели тренировончый набор данных из 25 различно написанных цифр

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
plt.show()
"""

# Переопределили глубину цвета (RGB - 3, а наше MNIST - 1). Требуется объявить это явно
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
# Последняя в кортеже цифра 1 означает, что изображения черно-белые (глубина цвета = 1)

# Преобразовываем значение каждой метки в массив из 10 элементов, состоящих из 0 и 1
# Положение единицы соответствует значениям метки

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('Метки тренировочного набора данных после преобразования', y_train.shape)
print('Метки тестового набора данных после преобразования', y_test.shape)

"""
print('Таблица тестовых меток')
for i in range(25):
    y = y_train[i]
    print(y)
    i = i + 1
"""
""" Последовательная модель Sequential"""

"""
# Создание модели
model = Sequential()
# Первый сверточный слой
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
# Второй сверточный слой
model.add(Conv2D(32, kernel_size=3, activation='relu'))
# Создаем вектор для полносвязной сети
model.add(Flatten())
# Создаем однослойный персептрон
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1)
print(hist.history)

# Построение графика точности предсказания
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Точность модели')
plt.ylabel('Точность')
plt.xlabel('Эпохи')
plt.legend(['Учебные', 'Тестовые'], loc='upper left')
plt.show()

# Построение графика потерь (ошибок)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Потери модели')
plt.ylabel('Потери')
plt.xlabel('Эпохи')
plt.legend(['Учебные', 'Тестовые'], loc='upper left')
plt.show()
"""
# Второй вариант сети со сходимостью
model = Sequential()

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D())
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)
# Построение графика точности предсказания"""

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Точность модели')
plt.ylabel('Точность')
plt.xlabel('Эпохи')
plt.legend(['Учебные', 'Тестовые'], loc='upper left')
plt.show()

# Построение графика потерь (ошибок)"""

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Потери модели')
plt.ylabel('Потери')
plt.xlabel('Эпохи')
plt.legend(['Учебные', 'Тестовые'], loc='upper left')
plt.show()

""" Загружаем нейронку в файл и проводим испытание """
# Запись полученной сети в файл
# Это моя рабочая директория данного проекта, используйте свой адресс
path = "D:\c++\PyCharm\pythonProject\keras_theano_tensorflow\my_model_1.h5"
model.save(path)

# Удаление модели
del model
# Подгрузка обученной модели сети из файла
model = load_model(path)

# Загрузка обученной модели сети из файла
model_New = load_model(path)
# y_train_pr = model_New.predict_classes(X_train[:3], verbose=0) --------> с 01.01.2021 не работает
# рабочая схема предсказания ниже
predict = model_New.predict(X_train, verbose=0)
predict_cl = np.argmax(predict, axis=1)


print('Первые три символа:\n', y_train[:3])
print('Первые три предсказания', predict_cl[:3])
