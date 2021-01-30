import os
import logging

# Environment Variables

##Supresses tensorflow warnings and errors, change to 2 to show errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from thesis.models.InceptionV3 import inceptionV3_compile
from thesis.data import data_reader_augmentation_train


trainGenerator = data_reader_augmentation_train('data/i2a2-brasil-pneumonia-classification')

model = inceptionV3_compile()


model.fit(trainGenerator,
           epochs=5,
           verbose=1,
           batch_size=32)

model.save_weights("./InceptionV3_weights.hdf5")