import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ["Футболка/топ", "Штаны", "Толстовка", "Платье", "Пальто", "Туфли", "Рубашка", "Кроссовки", "Сумка", "Ботинки"]

# plt.clf()
# plt.cla
# plt.figure(1)
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
train_images = train_images / 255.0
test_images = test_images / 255.0
#plt.show()QQQ2

# plt.figure(2, figsize=(10,10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(128, activation = tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax)
    ])

model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = "sparse_categorical_crossentropy",
              metrics = ["accuracy"])

model.fit(train_images, train_labels, epochs = 5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test acc: ", test_acc)

predictions = model.predict(test_images)
predictions[0]

np.argmax(predictions[0])
test_labels[0]

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap = plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color = color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color = "#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color("red")
    thisplot[true_label].set_color("blue")

# i = 0
# plt.figure(figsize = (6, 3))
# plt.subplot(1, 2, 1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1, 2, 2)
# plot_value_array(i, predictions, test_labels)
# plt.xticks(range(10), class_names, rotation = 45)
# plt.show()

num_rows = 4
num_cols = 3
num_images = num_rows * num_cols
x = random.randrange(9999 - num_images)
plt.figure(figsize = (2 * 2 * num_cols, 2 * num_rows))
for i in range(x, num_images + x):
    plt.subplot(num_rows, 2 * num_cols, 2 * (i - x) + 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * (i - x) + 2)
    plot_value_array(i, predictions, test_labels)
    #plt.xticks(range(10), class_names, rotation = 45)
img = test_images[0]
img = (np.expand_dims(img, 0))
predictions_single = model.predict(img)
print(predictions_single)
plot_value_array(0, predictions_single, test_labels)
plt.show()