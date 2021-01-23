from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,GlobalMaxPooling2D
import tensorflow as tf


def inceptionV3_compile():


    pretrained_model = InceptionV3(include_top=False,
                                weights='imagenet',
                                input_shape=(224,224,3))

    model = Sequential()
    model.add(pretrained_model)
    model.add(GlobalMaxPooling2D())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=opt,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    model.summary()

    return model