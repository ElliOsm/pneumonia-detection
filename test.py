from thesis.models.ResNet50 import resnet50_compile
from thesis.models.InceptionV3 import inceptionV3_compile
from thesis.data import data_reader_augmentation


trainGenerator, validationGenerator = data_reader_augmentation('data/i2a2-brasil-pneumonia-classification')

#model = resnet50_compile()
model = inceptionV3_compile()

model.fit(trainGenerator,
           epochs=10,
           verbose=1,
           batch_size=32,
           validation_data=validationGenerator)


predict = model.predict(trainGenerator,
                        verbose=1)
