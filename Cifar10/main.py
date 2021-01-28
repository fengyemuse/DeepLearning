import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import backend
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard

backend.set_image_data_format('channels_first')

batch_size = 128
epochs = 200
iterations = 391
num_classes = 10
dropout = 0.5
log_filepath = r'.\nin'


def normalize_preprocessing(x_train, x_validation):
    x_train = x_train.astype('float32')
    x_validation = x_validation.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_validation[:, :, :, i] = (x_validation[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_validation


def scheduler(epoch):
    if epoch <= 60:
        return 0.05
    if epoch <= 120:
        return 0.01
    if epoch <= 160:
        return 0.002
    return 0.0004


def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(192, (5, 5),
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                     kernel_initializer=RandomNormal(stddev=0.01),
                                     input_shape=x_train.shape[1:],
                                     activation='relu'))
    model.add(tf.keras.layers.Conv2D(160, (1, 1),
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                     kernel_initializer=RandomNormal(stddev=0.05),
                                     activation='relu'))
    model.add(tf.keras.layers.Conv2D(96, (1, 1),
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                     kernel_initializer=RandomNormal(stddev=0.05),
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Conv2D(192, (5, 5),
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                     kernel_initializer=RandomNormal(stddev=0.05),
                                     activation='relu'))
    model.add(tf.keras.layers.Conv2D(192, (1, 1),
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                     kernel_initializer=RandomNormal(stddev=0.05),
                                     activation='relu'))
    model.add(tf.keras.layers.Conv2D(192, (1, 1),
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                     kernel_initializer=RandomNormal(stddev=0.05),
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Conv2D(192, (3, 3),
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                     kernel_initializer=RandomNormal(stddev=0.05),
                                     activation='relu'))
    model.add(tf.keras.layers.Conv2D(192, (1, 1),
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                     kernel_initializer=RandomNormal(stddev=0.05),
                                     activation='relu'))
    model.add(tf.keras.layers.Conv2D(10, (1, 1),
                                     padding='same',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                     kernel_initializer=RandomNormal(stddev=0.05),
                                     activation='relu'))
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    (x_train, y_train), (x_validation, y_validation) = cifar10.load_data()

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_validation = tf.keras.utils.to_categorical(y_validation, num_classes)
    x_train, x_validation = normalize_preprocessing(x_train, x_validation)
    model = build_model()
    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr, tb_cb]
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=cbks,
              validation_data=(x_validation, y_validation),
              verbose=2)
    model.save('nin.h5')
