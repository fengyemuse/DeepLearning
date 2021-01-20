from sklearn import datasets
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt

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
history = Model.fit(x, Y_labels, validation_split=0.2, epochs=200, batch_size=5, verbose=0)
scores = Model.evaluate(x, Y_labels, verbose=0)
print('%s:  %f %%' % (Model.metrics_names[1], scores[1] * 100))
print(history.history.keys())
plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
