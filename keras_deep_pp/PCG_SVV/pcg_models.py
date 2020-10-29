from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D,Dense,LSTM,Dropout,BatchNormalization,Activation,GlobalAveragePooling1D

import keras



def semi_sv_vggnet10001c():
    # Design model
    model = Sequential()
    model.add(Conv1D(64, 12, activation='relu',
                            input_shape=(1000,1)))
    model.add(Conv1D(64, 12, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2,2))
    model.add(Conv1D(128, 12, activation='relu'))
    model.add(Conv1D(128, 12, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2,2))
    model.add(Conv1D(128, 8, activation='relu'))
    model.add(Conv1D(128, 8, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2,2))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2,2))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2,2))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2,2))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2,2))
    model.add(keras.layers.Dropout(0.5))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))

    return model

def pcg_SV_testmodel01():
    # Design model
    model = Sequential()
    model.add(Conv1D(128, 32, activation='relu',strides=2,
                     input_shape=(3000, 2)))


    model.add(Conv1D(128, 32, activation='relu'))
    model.add(Conv1D(128, 24, activation='relu'))
    model.add(Conv1D(128, 24, strides=2, activation='relu'))

    model.add(Conv1D(128, 12, activation='relu'))
    model.add(Conv1D(128, 12,strides=2, activation='relu'))
    model.add(Conv1D(256, 12, activation='relu'))
    model.add(Conv1D(256, 12,strides=2, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8,strides=2, activation='relu'))
    model.add(Conv1D(512, 8, activation='relu'))
    model.add(Conv1D(512, 8,strides=2, activation='relu'))
    model.add(Conv1D(512, 8, activation='relu'))
    model.add(Conv1D(512, 8,strides=2, activation='relu'))
    model.add(Conv1D(1024, 3, activation='relu'))
    model.add(Conv1D(1024, 3,strides=2, activation='relu'))
    model.add(keras.layers.Dropout(0.05))
    # model.add(layers.Dense(512, activation='relu'))
    #model.add(Dense(4096, activation='relu'))
    #model.add(Dense(4096, activation='relu'))
    #model.add(Dense(1000, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))
    return model



def pcg_SVV_testmodel01():
    # Design model
    model = Sequential()
    model.add(Conv1D(128, 64, activation='relu',strides=2,
                     input_shape=(10000, 2)))


    model.add(Conv1D(128, 32, activation='relu'))
    model.add(Conv1D(128, 24, activation='relu'))
    model.add(Conv1D(128, 24, strides=2, activation='relu'))

    model.add(Conv1D(128, 12, activation='relu'))
    model.add(Conv1D(128, 12,strides=2, activation='relu'))
    model.add(Conv1D(128, 12, activation='relu'))
    model.add(Conv1D(128, 12,strides=2, activation='relu'))
    model.add(Conv1D(128, 8, activation='relu'))
    model.add(Conv1D(128, 8,strides=2, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8,strides=2, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8,strides=2, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(512, 3,strides=2, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(1024, 3,strides=2, activation='relu'))
    model.add(Conv1D(1024, 3, activation='relu'))
    model.add(Conv1D(2048, 3, strides=2,activation='relu'))
    model.add(Conv1D(2048, 3,activation='relu'))
    model.add(Conv1D(2048, 2, strides=2, activation='relu'))
    model.add(keras.layers.Dropout(0.05))
    # model.add(layers.Dense(512, activation='relu'))
    #model.add(Dense(4096, activation='relu'))
    #model.add(Dense(4096, activation='relu'))
    #model.add(Dense(1000, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))
    return model







def widecnn_pcg_testmodel02():
    # Design model
    model = Sequential()
    model.add(Conv1D(128, 500, activation='relu',
                     input_shape=(10000, 1),strides=2))
    model.add(Conv1D(128, 300, strides=2, activation='relu'))
    model.add(Conv1D(128, 200, strides=2, activation='relu'))
    model.add(Conv1D(128, 100,strides=2, activation='relu'))
    model.add(Conv1D(128, 12, activation='relu'))
    model.add(Conv1D(128, 12,strides=2, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8,strides=2, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(512, 3,strides=2, activation='relu'))
    model.add(Conv1D(1024, 3,strides=2, activation='relu'))
    model.add(Conv1D(1024, 3, activation='relu'))
    model.add(Conv1D(2048, 3, strides=2,activation='relu'))
    model.add(Conv1D(2048, 3,activation='relu'))
    model.add(Conv1D(2048, 2, strides=2, activation='relu'))
    model.add(Conv1D(2048, 2,  activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(keras.layers.Dropout(0.05))
    # model.add(layers.Dense(512, activation='relu'))

    #model.add(Dense(4096, activation='relu'))
    #model.add(Dense(4096, activation='relu'))
    #model.add(Dense(1000, activation='relu'))
    #model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))
    return model





def widecnn_3000():
    # Design model
    model = Sequential()
    model.add(Conv1D(128, 200, activation='relu',strides=2,
                     input_shape=(3000, 1)))

    model.add(Conv1D(128, 24, activation='relu'))
    model.add(Conv1D(128, 150, strides=2, activation='relu'))

    model.add(Conv1D(128, 12, activation='relu'))
    model.add(Conv1D(128, 100,strides=2, activation='relu'))
    model.add(Conv1D(128, 12, activation='relu'))
    model.add(Conv1D(128, 12,strides=2, activation='relu'))
    model.add(Conv1D(128, 8, activation='relu'))
    model.add(Conv1D(128, 8,strides=2, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8,strides=2, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8,strides=2, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(1024, 1, activation='relu'))
    model.add(keras.layers.Dropout(0.05))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))
    return model


def widecnn_pcg_vggnet1ch():
    # Design model
    model = Sequential()
    model.add(Conv1D(128, 64, activation='relu',
                     input_shape=(10000, 1)))


    model.add(Conv1D(128, 32, strides=2, activation='relu'))
    model.add(Conv1D(128, 24, activation='relu'))
    model.add(Conv1D(128, 24, strides=2, activation='relu'))

    model.add(Conv1D(128, 12, activation='relu'))
    model.add(Conv1D(128, 12,strides=2, activation='relu'))
    model.add(Conv1D(128, 12, activation='relu'))
    model.add(Conv1D(128, 12,strides=2, activation='relu'))
    model.add(Conv1D(128, 8, activation='relu'))
    model.add(Conv1D(128, 8,strides=2, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8,strides=2, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8,strides=2, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(512, 3,strides=2, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(1024, 3,strides=2, activation='relu'))
    model.add(Conv1D(1024, 3, activation='relu'))
    model.add(Conv1D(2048, 3, strides=2,activation='relu'))
    model.add(Conv1D(2048, 3,activation='relu'))
    model.add(Conv1D(2048, 2, strides=2, activation='relu'))
    model.add(keras.layers.Dropout(0.05))
    # model.add(layers.Dense(512, activation='relu'))

    #model.add(Dense(4096, activation='relu'))
    #model.add(Dense(4096, activation='relu'))
    #model.add(Dense(1000, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))
    return model


def widecnn_pcg_vggnet():
    # Design model
    model = Sequential()
    model.add(Conv1D(128, 64, activation='relu',
                     input_shape=(10000, 5)))


    model.add(Conv1D(128, 32, strides=2, activation='relu'))
    model.add(Conv1D(128, 24, activation='relu'))
    model.add(Conv1D(128, 24, strides=2, activation='relu'))

    model.add(Conv1D(128, 12, activation='relu'))
    model.add(Conv1D(128, 12,strides=2, activation='relu'))
    model.add(Conv1D(128, 12, activation='relu'))
    model.add(Conv1D(128, 12,strides=2, activation='relu'))
    model.add(Conv1D(128, 8, activation='relu'))
    model.add(Conv1D(128, 8,strides=2, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8,strides=2, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8,strides=2, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(512, 3,strides=2, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(1024, 3,strides=2, activation='relu'))
    model.add(Conv1D(1024, 3, activation='relu'))
    model.add(Conv1D(2048, 3, strides=2,activation='relu'))
    model.add(Conv1D(2048, 3,activation='relu'))
    model.add(Conv1D(2048, 2, strides=2, activation='relu'))
    model.add(keras.layers.Dropout(0.05))
    # model.add(layers.Dense(512, activation='relu'))

    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))
    return model



def widecnn_pp_600_batch():
    # Design model
    model = Sequential()
    model.add(Conv1D(128, 12,activation='relu',
                            input_shape=(600,5)))
    model.add(Conv1D(128, 12,strides=2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(128, 12))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(128, 12,strides=2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(256, 8))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(256, 8,strides=2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(256, 8))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(256, 8,strides=2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(512, 5))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(512, 5,strides=2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(512, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv1D(512, 3,strides=3))

    model.add(keras.layers.Dropout(0.05))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))

    return model



def widecnn_pp_600_vggnet():
    # Design model
    model = Sequential()
    model.add(Conv1D(128, 12,activation='relu',
                            input_shape=(600,5)))
    model.add(Conv1D(128, 12,strides=2, activation='relu'))
    model.add(Conv1D(128, 12, activation='relu'))
    model.add(Conv1D(128, 12,strides=2, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8,strides=2, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8,strides=2, activation='relu'))
    model.add(Conv1D(512, 5, activation='relu'))
    model.add(Conv1D(512, 5,strides=2, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(512, 3,strides=3, activation='relu'))
    model.add(keras.layers.Dropout(0.05))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))


def widecnn_pp_600():
    # Design model
    model = Sequential()
    model.add(Conv1D(128, 12,activation='relu',
                            input_shape=(600,5)))
    model.add(Conv1D(128, 12,strides=2, activation='relu'))
    model.add(Conv1D(128, 12, activation='relu'))
    model.add(Conv1D(128, 12,strides=2, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8,strides=2, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8,strides=2, activation='relu'))
    model.add(Conv1D(512, 5, activation='relu'))
    model.add(Conv1D(512, 5,strides=2, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(512, 3,strides=3, activation='relu'))
    model.add(keras.layers.Dropout(0.05))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(keras.layers.Flatten())

    model.add(Dense(1, activation='linear'))


    model.add(Dense(1, activation='linear'))

    return model



def widecnn_pcg20000():
    # Design model
    model = Sequential()
    model.add(Conv1D(128, 128, strides=5, activation='relu',
                            input_shape=(20000,5)))
    model.add(Conv1D(128, 12, activation='relu'))
    model.add(Conv1D(128, 64,strides=2, activation='relu'))
    model.add(Conv1D(128, 12, activation='relu'))
    model.add(Conv1D(128, 32,strides=2, activation='relu'))
    model.add(Conv1D(128, 8, activation='relu'))
    model.add(Conv1D(128, 8,strides=2, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8,strides=2, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8,strides=2, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(512, 3,strides=2, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(1024, 3,strides=2, activation='relu'))
    model.add(Conv1D(1024, 3, activation='relu'))
    model.add(Conv1D(1024, 3, activation='relu'))
    model.add(Conv1D(2048, 3, strides=5, activation='relu'))
    model.add(Conv1D(2048, 3, strides=2 ,activation='relu'))
    #model.add(Conv1D(2048, 3, activation='relu'))
    model.add(keras.layers.Dropout(0.05))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(keras.layers.Flatten())
    #model.add(Dense(1000, activation='relu'))
    model.add(Dense(1, activation='linear'))

    return model


def widecnn_pcg_full():
    # Design model
    model = Sequential()
    model.add(Conv1D(128, 32, activation='relu',
                     input_shape=(10000, 5)))


    model.add(Conv1D(128, 32, strides=2, activation='relu'))
    model.add(Conv1D(128, 24, activation='relu'))
    model.add(Conv1D(128, 24, strides=2, activation='relu'))

    model.add(Conv1D(128, 12, activation='relu'))
    model.add(Conv1D(128, 12,strides=2, activation='relu'))
    model.add(Conv1D(128, 12, activation='relu'))
    model.add(Conv1D(128, 12,strides=2, activation='relu'))
    model.add(Conv1D(128, 8, activation='relu'))
    model.add(Conv1D(128, 8,strides=2, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8,strides=2, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8,strides=2, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(512, 3,strides=2, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(1024, 3,strides=2, activation='relu'))
    model.add(Conv1D(1024, 3, activation='relu'))
    model.add(Conv1D(2048, 3, strides=2,activation='relu'))
    model.add(Conv1D(2048, 3,activation='relu'))
    model.add(Conv1D(2048, 2, strides=2, activation='relu'))
    model.add(keras.layers.Dropout(0.05))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))
    return model

def widecnn_pcg():
    # Design model
    model = Sequential()
    model.add(Conv1D(128, 12, strides=5, activation='relu',
                            input_shape=(10000,5)))
    model.add(Conv1D(128, 12, activation='relu'))
    model.add(Conv1D(128, 12,strides=2, activation='relu'))
    model.add(Conv1D(128, 12, activation='relu'))
    model.add(Conv1D(128, 12,strides=2, activation='relu'))
    model.add(Conv1D(128, 8, activation='relu'))
    model.add(Conv1D(128, 8,strides=2, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8,strides=2, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8,strides=2, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(512, 3,strides=2, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(1024, 3,strides=2, activation='relu'))
    model.add(Conv1D(1024, 3, activation='relu'))
    model.add(Conv1D(1024, 3, activation='relu'))
    model.add(Conv1D(2048, 3, strides=5, activation='relu'))
    model.add(keras.layers.Dropout(0.05))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))

    return model


def semi_vggnet():
    # Design model
    model = Sequential()
    model.add(Conv1D(64, 12, activation='relu',
                            input_shape=(10000,5)))
    model.add(Conv1D(64, 12, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2,2))
    model.add(Conv1D(64, 12, activation='relu'))
    model.add(Conv1D(64, 12, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2,2))
    model.add(Conv1D(128, 12, activation='relu'))
    model.add(Conv1D(128, 12, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2,2))
    model.add(Conv1D(128, 8, activation='relu'))
    model.add(Conv1D(128, 8, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2,2))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2,2))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2,2))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2,2))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2,2))
    #model.add(keras.layers.Dropout(0.))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))

    return model

def semi_vggnet1000():
    # Design model
    model = Sequential()
    model.add(Conv1D(64, 12, activation='relu',
                            input_shape=(1000,3)))
    model.add(Conv1D(64, 12, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2,2))
    model.add(Conv1D(128, 12, activation='relu'))
    model.add(Conv1D(128, 12, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2,2))
    model.add(Conv1D(128, 8, activation='relu'))
    model.add(Conv1D(128, 8, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2,2))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2,2))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2,2))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2,2))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2,2))
    model.add(keras.layers.Dropout(0.5))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))

    return model

def test01_model():
    model = Sequential()
    model.add(Conv1D(128, 12, activation='relu',
                     input_shape=(2000, 3)))
    model.add(keras.layers.AveragePooling1D(2, 2))
    model.add(Conv1D(128, 12, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2, 2))
    model.add(Conv1D(128, 12, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2, 2))
    model.add(Conv1D(128, 8, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2, 2))
    model.add(Conv1D(64, 8, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2, 2))
    model.add(Conv1D(64, 8, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2, 2))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2, 2))
    model.add(Conv1D(16, 3, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2, 2))
    model.add(Conv1D(8, 3, activation='relu'))
    model.add(keras.layers.AveragePooling1D(2, 2))
    model.add(keras.layers.Dropout(0.5))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))
    return model

