from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from matplotlib import pyplot as plt
import numpy as np

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 为保证公平起见，两种方式都使用相同的随机种子
np.random.seed(7)
batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# without_BN模型的训练
model_without_bn = Sequential()
model_without_bn.add(Conv2D(32, (3, 3), padding='same',
                            input_shape=x_train.shape[1:]))
model_without_bn.add(Activation('relu'))
model_without_bn.add(Conv2D(32, (3, 3)))
model_without_bn.add(Activation('relu'))
model_without_bn.add(MaxPooling2D(pool_size=(2, 2)))
model_without_bn.add(Dropout(0.25))

model_without_bn.add(Conv2D(64, (3, 3), padding='same'))
model_without_bn.add(Activation('relu'))
model_without_bn.add(Conv2D(64, (3, 3)))
model_without_bn.add(Activation('relu'))
model_without_bn.add(MaxPooling2D(pool_size=(2, 2)))
model_without_bn.add(Dropout(0.25))

model_without_bn.add(Flatten())
model_without_bn.add(Dense(512))
model_without_bn.add(Activation('relu'))
model_without_bn.add(Dropout(0.5))
model_without_bn.add(Dense(num_classes))
model_without_bn.add(Activation('softmax'))

opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

model_without_bn.compile(loss='categorical_crossentropy',
                         optimizer=opt,
                         metrics=['accuracy'])
if not data_augmentation:
    history_without_bn = model_without_bn.fit(x_train, y_train,
                                              batch_size=batch_size,
                                              epochs=epochs,
                                              validation_data=(x_test, y_test),
                                              shuffle=True)
else:
    # 使用数据增强获取更多的训练数据
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    datagen.fit(x_train)
    history_without_bn = model_without_bn.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                                        epochs=epochs,
                                                        validation_data=(x_test, y_test), workers=4)

# with_BN模型的训练
model_with_bn = Sequential()
model_with_bn.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model_with_bn.add(BatchNormalization())
model_with_bn.add(Activation('relu'))
model_with_bn.add(Conv2D(32, (3, 3)))
model_with_bn.add(BatchNormalization())
model_with_bn.add(Activation('relu'))
model_with_bn.add(MaxPooling2D(pool_size=(2, 2)))

model_with_bn.add(Conv2D(64, (3, 3), padding='same'))
model_with_bn.add(BatchNormalization())
model_with_bn.add(Activation('relu'))
model_with_bn.add(Conv2D(64, (3, 3)))
model_with_bn.add(BatchNormalization())
model_with_bn.add(Activation('relu'))
model_with_bn.add(MaxPooling2D(pool_size=(2, 2)))

model_with_bn.add(Flatten())
model_with_bn.add(Dense(512))
model_with_bn.add(BatchNormalization())
model_with_bn.add(Activation('relu'))
model_with_bn.add(Dense(num_classes))
model_with_bn.add(BatchNormalization())
model_with_bn.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)

model_with_bn.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

if not data_augmentation:
    history_with_bn = model_without_bn.fit(x_train, y_train,
                                           batch_size=batch_size,
                                           epochs=epochs,
                                           validation_data=(x_test, y_test),
                                           shuffle=True)
else:
    # 使用数据增强获取更多的训练数据
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    datagen.fit(x_train)
    history_with_bn = model_with_bn.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs,
                                                  validation_data=(x_test, y_test), workers=4, use_multiprocessing=True)

# 比较两种模型的精确度
plt.plot(history_without_bn.history['val_acc'])
plt.plot(history_with_bn.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Validation Accuracy')
plt.xlabel('Epoch')
plt.legend(['No Batch Normalization', 'With Batch Normalization'], loc='lower right')
plt.grid(True)
plt.show()

# 比较两种模型的损失率
plt.plot(history_without_bn.history['loss'])
plt.plot(history_without_bn.history['val_loss'])
plt.plot(history_with_bn.history['loss'])
plt.plot(history_with_bn.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(
    ['Training Loss without BN', 'Validation Loss without BN', 'Training Loss with BN', 'Validation Loss with BN'],
    loc='upper right')
plt.show()
