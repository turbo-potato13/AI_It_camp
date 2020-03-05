#Подключаем библиотеки
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#Импорт из tensorflow и загрузка данных
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Сами прописываем имена классов
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#Вывод одного изображения
plt.clf()
plt.cla
plt.figure(1)
plt.imshow(train_images[0])# номер модели, любой от 0 до 599999
plt.colorbar()
plt.grid(False)
#Масштабируем изображение от 0 до 1
train_images = train_images / 255.0
test_images = test_images / 255.0
#plt.show()# выводит изображение

#Выводим первые 25 изображений из тренировочного
plt.figure(figsize=(10,10))#размер картинки
for i in range(25):# колличество картинок
    plt.subplot(5,5,i+1)#Счетчик 5х5
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
#plt.show()

#Построение слоев нейронной сети
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),#Преобразование из 2d-массива (28 на 28 пикселей) в 1d-массив из 28 * 28 = 784 пикселей
    keras.layers.Dense(128, activation=tf.nn.relu),#содержит 128 узлов (или нейронов)
    keras.layers.Dense(10, activation=tf.nn.softmax)#это слой с 10 узлами tf.nn.softmax , который возвращает массив из десяти вероятностных оценок, сумма которых равна 1
])#Каждый узел содержит оценку, которая указывает вероятность того, что текущее изображение принадлежит одному из 10 классов.

#Компиляция модели
model.compile(optimizer=tf.train.AdamOptimizer(),# это то, как модель обновляется на основе данных, которые она видит, и функции потери
              loss='sparse_categorical_crossentropy',#измеряет насколько точная модель во время обучения
              metrics=['accuracy'])#используется для контроля за этапами обучения и тестирования

#Обучение модели
model.fit(train_images, train_labels, epochs=5)#Подача данных обучения модели(train_images и train_labels)
#При моделировании модели отображаются показатели потерь (loss) и точности (acc).

#Проверка модели на тестовых данных
test_loss, test_acc = model.evaluate(test_images, test_labels)#
print('Test accuracy:', test_acc)

#Прогнозирование
predictions = model.predict(test_images)#Используем модель для прогнозирования некоторых изображений.
predictions[0]#Здесь модель предсказала метку для каждого изображения в тестовом наборе. Давайте посмотрим на первое предсказание:

np.argmax(predictions[0]) #9
test_labels[0]  #9


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)



def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
