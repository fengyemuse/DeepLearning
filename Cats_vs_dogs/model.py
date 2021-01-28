import tensorflow as tf


def create_model():
    cnn_model = tf.keras.Sequential()
    cnn_model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    cnn_model.add(tf.keras.layers.MaxPool2D((2, 2)))
    cnn_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(tf.keras.layers.MaxPool2D((2, 2)))
    cnn_model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    cnn_model.add(tf.keras.layers.MaxPool2D((2, 2)))
    cnn_model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    cnn_model.add(tf.keras.layers.MaxPool2D((2, 2)))
    cnn_model.add(tf.keras.layers.Flatten())
    cnn_model.add(tf.keras.layers.Dropout(0.5))
    cnn_model.add(tf.keras.layers.Dense(512, activation='relu'))
    cnn_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    cnn_model.compile(loss=tf.keras.losses.binary_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])
    cnn_model.summary()
    return cnn_model


if __name__ == '__main__':
    CNN_model = create_model()
