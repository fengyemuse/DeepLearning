import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

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


train_dir = r'D:\work\Deeplearning_Data\cats_dogs\cats_vs_dogs_data\train'
validation = r'D:\work\Deeplearning_Data\cats_dogs\cats_vs_dogs_data\validation'

if __name__ == '__main__':
    CNN_model = create_model()
    # train_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    # test_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
    validation_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
    history = CNN_model.fit_generator(generator=train_generator, steps_per_epoch=100,
                                      epochs=30,
                                      validation_data=validation_generator,
                                      validation_steps=50)
    CNN_model.save('cats_and_dogs_1.h5')
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
