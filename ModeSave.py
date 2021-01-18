from sklearn import datasets
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


dataset = datasets.load_iris()
x = dataset.data
y = dataset.target
Y_labels = to_categorical(y, num_classes=3)  # one hot编码
seed = 7
np.random.seed(seed)


def create_model(optimizer='rmsprop'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(6, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


Model = create_model()
Model.fit(x, Y_labels, epochs=300, batch_size=5, verbose=0)
scores = Model.evaluate(x, Y_labels, verbose=0)
print('%s: %f%%' % (Model.metrics_names[1], scores[1] * 100))

model_json = Model.to_json()
with open('model.json', 'w') as file:
    file.write(model_json)

Model.save_weights('model.json.h5')
