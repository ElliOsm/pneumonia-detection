import pandas as pd
import numpy as np

from keras_preprocessing.image import ImageDataGenerator


def train_data_reader(dir):
    image_directory = dir + '/images'
    data = pd.read_csv(dir + '/train.csv')

    dataGenerator = ImageDataGenerator()

    trainGenerator = dataGenerator.flow_from_dataframe(data,
                                                       directory=image_directory,
                                                       x_col='fileName',
                                                       y_col='pneumonia',
                                                       target_size=(244, 244),
                                                       color_mode='rgb',
                                                       class_mode='raw',
                                                       shuffle=False,
                                                       batch_size=128)

    return trainGenerator


def data_reader_augmentation_train(dir):
    image_directory = dir + '/images'
    data = pd.read_csv(dir + '/train.csv')

    # map = {
    #     0 : np.array([1,0]),
    #     1 : np.array([0,1])
    # }
    #
    # data["pneumonia"] = data["pneumonia"].apply(lambda x : map[x])
    #
    # print(data["pneumonia"].head)

    dataGenerator = ImageDataGenerator(rescale=1.0 / 255,
                                       rotation_range=40,
                                       width_shift_range=0.20,
                                       height_shift_range=0.20,
                                       brightness_range=[1.0, 1.5],
                                       zoom_range=[1.0, 1.2],
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       validation_split=0.2)

    trainGenerator = dataGenerator.flow_from_dataframe(data,
                                                       directory=image_directory,
                                                       x_col='fileName',
                                                       y_col='pneumonia',
                                                       target_size=(244, 244),
                                                       color_mode='rgb',
                                                       class_mode='raw',
                                                       shuffle=False,
                                                       batch_size=32,
                                                       subset='training')


    return trainGenerator


def data_reader_augmentation_valid(dir):
    image_directory = dir + '/images'
    data = pd.read_csv(dir + '/train.csv')
    # validation =

    dataGenerator = ImageDataGenerator(rescale=1.0 / 255,
                                       rotation_range=40,
                                       width_shift_range=0.20,
                                       height_shift_range=0.20,
                                       brightness_range=[1.0, 1.5],
                                       zoom_range=[1.0, 1.2],
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       validation_split=0.2)

    validationGenerator = dataGenerator.flow_from_dataframe(data,
                                                            directory=image_directory,
                                                            x_col='fileName',
                                                            y_col='pneumonia',
                                                            target_size=(244, 244),
                                                            color_mode='rgb',
                                                            class_mode='raw',
                                                            shuffle=False,
                                                            batch_size=32,
                                                            subset='validation')
    return validationGenerator
