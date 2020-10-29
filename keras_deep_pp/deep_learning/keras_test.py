import keras
from utils.my_classes import DataGenerator
from keras import models
from keras import layers
import numpy as np
from keras import optimizers


def cnn2d_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(20000, 100,1)))
    model.add(layers.MaxPooling2D((5,2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((5, 1)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    #model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))

    return model
# Parameters
params = {'dim': (20000,100),
          'batch_size': 1,
          'n_classes': 1,
          'n_channels': 1,
          'shuffle': True}

y=[1,2.0,3]

y = np.array([0, 2, 1.5, 2, 0])
keras.utils.to_categorical(y, num_classes=3)



a = ['180313_0_1','180313_0_2']
# Datasets
import os
data = os.listdir('data/')
partition = {'train' : a , 'validation' : ['180313_0_0']}
#labels = {'180313_0_1' : 57.08444, '180313_0_2' :75.627174, '180313_0_0' : 73.48143 }
labels = {'180313_0_1' : 1, '180313_0_2' :0, '180313_0_0' : 1 }
#[57.08444 , 75.627174, 73.48143]

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

# Design model
model = cnn2d_model()
[...] # Architecture
opt = optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='mse', optimizer=opt, metrics=['mse'])


# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    workers=1)

