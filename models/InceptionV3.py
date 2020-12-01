from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf


def inceptionV3_compile():


    pretrained_model = InceptionV3(include_top=True,
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