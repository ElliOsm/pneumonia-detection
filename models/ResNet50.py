from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf


def resnet50_compile():


    pretrained_model = ResNet50(include_top=True,
                                weights='imagenet',
                                input_tensor=None,
                                input_shape=None,
                                pooling=None)


    new_model = Sequential()
    new_model.add(pretrained_model)
    new_model.add(Dense(1, activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    new_model.compile(optimizer=opt,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    return new_model