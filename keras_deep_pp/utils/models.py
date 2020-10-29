from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D,Dense,LSTM,Dropout,Reshape,GlobalAveragePooling1D
import keras
from keras.layers import Activation, Lambda
from keras.layers import Conv1D, SpatialDropout1D
from keras.models import Input, Model
import tensorflow as tf
import numpy as np
from keras import backend as K

def svv_wcnn_model01():
    # Design model
    model = Sequential()
    model.add(Conv1D(64, 200, activation='relu',strides=2,
                            input_shape=(1000,2)))
    model.add(Conv1D(64, 12, activation='relu'))
    model.add(Conv1D(128, 100,strides=2, activation='relu'))
    model.add(Conv1D(128, 12, activation='relu'))
    model.add(Conv1D(128, 50,strides=2, activation='relu'))
    model.add(Conv1D(128, 8, activation='relu'))
    model.add(Conv1D(256, 25,strides=2, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(512, 1, activation='relu'))
    #model.add(keras.layers.Dropout(0.05))
    # model.add(layers.Dense(512, activation='relu'))
    #model.add(keras.layers.Flatten())
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1, activation='linear'))

    return model



def rcnn_test_300():
    # Design model
    model = Sequential()
    model.add(Conv1D(64, 100, activation='elu',
                            input_shape=(300,1),strides=3))
    model.add(Conv1D(256, 30, activation='elu',strides=2))
    #model.add(Reshape((1,model.output_shape[1])))
    model.add(LSTM(64,return_sequences=True))
    model.add(keras.layers.Dropout(0.05))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))
    return model


def rcnn_test03():
    # Design model
    model = Sequential()
    model.add(Conv1D(64, 100, activation='elu',
                            input_shape=(1000,3),strides=5))
    model.add(Conv1D(256, 8, activation='elu'))
    model.add(Dense(512, activation='elu'))
    model.add(Dense(512, activation='elu'))
    #model.add(Reshape((1,model.output_shape[1])))
    model.add(LSTM(256,return_sequences=True))
    model.add(keras.layers.Dropout(0.05))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))
    return model


def rcnn_test01():
    # Design model
    model = Sequential()
    model.add(Conv1D(64, 100, activation='elu',
                            input_shape=(1000,3),strides=2))
    model.add(Conv1D(128, 12, activation='elu'))
    model.add(Conv1D(128, 12,strides=2, activation='elu'))
    model.add(Conv1D(128, 8, activation='elu'))
    model.add(Conv1D(128, 8,strides=2, activation='elu'))
    model.add(Conv1D(256, 8, activation='elu'))
    model.add(Conv1D(256, 8,strides=2, activation='elu'))
    model.add(Conv1D(256, 8, activation='elu'))
    model.add(Conv1D(256, 8,strides=2, activation='elu'))
    model.add(Conv1D(512, 3, activation='elu'))
    model.add(Conv1D(512, 3,strides=2, activation='elu'))
    model.add(Conv1D(512, 3, activation='elu'))
    model.add(Conv1D(512, 3, activation='elu'))
    model.add(Conv1D(1024, 2, activation='elu'))
    model.add(keras.layers.Dropout(0.05))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))
    return model

def semi_vggnet1000_stride_1c():
    # Design model
    model = Sequential()
    model.add(Conv1D(64, 12, activation='relu',
                            input_shape=(300,1)))
    model.add(Conv1D(64, 12,strides=2, activation='relu'))
    model.add(Conv1D(128, 12, activation='relu'))
    model.add(Conv1D(128, 12,strides=2, activation='relu'))
    model.add(Conv1D(128, 8, activation='relu'))
    model.add(Conv1D(128, 8,strides=2, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8,strides=2, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(1024, 3, activation='relu',name='conv_end'))
    model.add(keras.layers.Dropout(0.05))
    #model.add(Dense(512, activation='relu')) # using cam
    model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))

    return modelz

def semi_vggnet1000_stride2c():
    # Design model
    model = Sequential()
    model.add(Conv1D(64, 12, activation='relu',
                            input_shape=(1000,2)))
    model.add(Conv1D(64, 12,strides=2, activation='relu'))
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
    model.add(Conv1D(1024, 3, activation='relu'))
    model.add(keras.layers.Dropout(0.05))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))

    return model


def semi_vggnet1000_stride1c():
    # Design model
    model = Sequential()
    model.add(Conv1D(64, 12, activation='relu',
                            input_shape=(1000,1)))
    model.add(Conv1D(64, 12,strides=2, activation='relu'))
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
    model.add(Conv1D(1024, 3, activation='relu'))
    model.add(keras.layers.Dropout(0.05))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))

    return model



def semi_vggnet1000_stride():
    # Design model
    model = Sequential()
    model.add(Conv1D(64, 12, activation='relu',
                            input_shape=(1000,3)))
    model.add(Conv1D(64, 12,strides=2, activation='relu'))
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
    model.add(Conv1D(1024, 3, activation='relu'))
    model.add(keras.layers.Dropout(0.05))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))

    return model



def semi_svv_ffttest():
    # Design model
    model = Sequential()
    model.add(Conv1D(64, 12, activation='relu',
                            input_shape=(1000,1)))
    model.add(Conv1D(64, 12,strides=2, activation='relu'))
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
    model.add(Conv1D(1024, 3, activation='relu'))
    model.add(keras.layers.Dropout(0.05))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))

    return model


def wdcnn_vggnet():
    # Design model
    model = Sequential()
    model.add(Conv1D(64, 12, activation='relu',
                            input_shape=(1000,3)))
    model.add(Conv1D(64, 12,strides=2, activation='relu'))
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
    model.add(Conv1D(1024, 3, activation='relu'))
    model.add(keras.layers.Dropout(0.05))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(1000,activation='relu'))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))

    return model

def wdcnn_fc1000():
    # Design model
    model = Sequential()
    model.add(Conv1D(64, 12, activation='relu',
                            input_shape=(1000,3)))
    model.add(Conv1D(64, 12,strides=2, activation='relu'))
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
    model.add(Conv1D(1024, 3, activation='relu'))
    model.add(keras.layers.Dropout(0.05))
    model.add(Dense(1000,activation='relu'))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))

    return model



def derivative_matrix (input_size):
    mat = np.zeros((input_size, input_size), dtype=np.float32)
    mat[1,0] = 1
    mat[0,0] = -1
    for i in range(1,input_size):
        mat[i,i] = 1
        mat[i-1,i] = -1
    return tf.constant(mat)


def cnn1d_moon():
    input_layer= Input(shape=(1000, 3))
    x = Conv1D(8, 12, activation='relu',padding='same')(input_layer)
    M1 = keras.layers.MaxPool1D(pool_size=2,strides=2,padding='same')(x)
    AVG1 =  keras.layers.AveragePooling1D(pool_size=2,strides=2,padding='same')(x)
    MAX1 = keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x)
    #DRV1 = tf.reshape(tf.matmul(tf.reshape(AVG1, [-1, 500]), derivative_matrix(500)), [-1, 500, 3])
    MIN1 = -tf.layers.max_pooling1d(inputs=-x, pool_size=2, strides=2, padding='same')
    MIN1 = Lambda(lambda m: -tf.layers.max_pooling1d(inputs=-x, pool_size=2, strides=2, padding='same'))


    x = keras.layers.concatenate([M1, MIN1],axis=2)

    #x = Conv1D(1024, 2, activation='relu')(x)
    #x = Conv1D(1024, 3, activation='relu')(x)
    #x = keras.layers.SpatialDropout1D(0.05)(x)
    # model.add(layers.Dense(512, activation='relu'))
    x = keras.layers.Flatten()(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    output_layer = x

    model = Model(input_layer, Output)

    return model



def model_cnn_1000 (X):
    with tf.variable_scope("model_cnn_1024/cnn", reuse=tf.AUTO_REUSE):
        # Input

        F0 = tf.placeholder(tf.float32, shape=[None, 1000, 3], name = 'InputWave')

        # L1 SigIn shape = (?, 1024, 1)
        L1 = tf.layers.conv1d(inputs=F0, filters=8, kernel_size=12, strides=1, padding='same', activation=tf.nn.relu, name='C1', reuse=tf.AUTO_REUSE)
        M1 = tf.layers.max_pooling1d(inputs=L1, pool_size=2, strides=2, padding='same')
        AVG1 = tf.layers.average_pooling1d(inputs=F0, pool_size=2, strides=2, padding='same')
        #DRV1 = tf.reshape(tf.matmul(tf.reshape(AVG1, [-1, 512]), derivative_matrix(512)), [-1, 512, 1])
        MAX1 = tf.layers.max_pooling1d(inputs=F0, pool_size=2, strides=2, padding='same')
        MIN1 = -tf.layers.max_pooling1d(inputs=-F0, pool_size=2, strides=2, padding='same')
        F1 = tf.concat([M1,MAX1,AVG1,MIN1],axis=2)
        # Conv -> (?, 512, 8)

        L2 = tf.layers.conv1d(inputs=F1, filters=8, kernel_size=12, strides=1, padding='same', activation=tf.nn.relu, name='C2', reuse=tf.AUTO_REUSE)
        M2 = tf.layers.max_pooling1d(inputs=L2, pool_size=2, strides=2, padding='same')
        AVG2 = tf.layers.average_pooling1d(inputs=AVG1, pool_size=2, strides=2, padding='same')
        #DRV2 = tf.reshape(tf.matmul(tf.reshape(AVG2, [-1, 256]), derivative_matrix(256)), [-1, 256, 1])
        MAX2 = tf.layers.max_pooling1d(inputs=MAX1, pool_size=2, strides=2, padding='same')
        MIN2 = -tf.layers.max_pooling1d(inputs=-MIN1, pool_size=2, strides=2, padding='same')
        F2 = tf.concat([M2,MAX2,AVG2,MIN2],axis=2)
        # Conv -> (?, 256, 8)

        L3 = tf.layers.conv1d(inputs=F2, filters=16, kernel_size=6, strides=1, padding='same', activation=tf.nn.relu, name='C3', reuse=tf.AUTO_REUSE)
        M3 = tf.layers.max_pooling1d(inputs=L3, pool_size=2, strides=2, padding='same')
        AVG3 = tf.layers.average_pooling1d(inputs=AVG2, pool_size=2, strides=2, padding='same')
        #DRV3 = tf.reshape(tf.matmul(tf.reshape(AVG3, [-1, 128]), derivative_matrix(128)), [-1, 128, 1])
        MAX3 = tf.layers.max_pooling1d(inputs=MAX2, pool_size=2, strides=2, padding='same')
        MIN3 = -tf.layers.max_pooling1d(inputs=-MIN2, pool_size=2, strides=2, padding='same')
        F3 = tf.concat([M3,MAX3,AVG3,MIN3],axis=2)
        # Conv -> (?, 128, 16)

        L4 = tf.layers.conv1d(inputs=F3, filters=16, kernel_size=6, strides=1, padding='same', activation=tf.nn.relu, name='C4', reuse=tf.AUTO_REUSE)
        M4 = tf.layers.max_pooling1d(inputs=L4, pool_size=2, strides=2, padding='same')
        AVG4 = tf.layers.average_pooling1d(inputs=AVG3, pool_size=2, strides=2, padding='same')
        #DRV4 = tf.reshape(tf.matmul(tf.reshape(AVG4, [-1, 64]), derivative_matrix(64)), [-1, 64, 1])
        MAX4 = tf.layers.max_pooling1d(inputs=MAX3, pool_size=2, strides=2, padding='same')
        MIN4 = -tf.layers.max_pooling1d(inputs=-MIN3, pool_size=2, strides=2, padding='same')
        F4 = tf.concat([M4,MAX4,AVG4,MIN4],axis=2)
        # Conv -> (?, 64, 16)

        L5 = tf.layers.conv1d(inputs=F4, filters=32, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu, name='C5', reuse=tf.AUTO_REUSE)
        M5 = tf.layers.max_pooling1d(inputs=L5, pool_size=2, strides=2, padding='same')
        AVG5 = tf.layers.average_pooling1d(inputs=AVG4, pool_size=2, strides=2, padding='same')
        #DRV5 = tf.reshape(tf.matmul(tf.reshape(AVG5, [-1, 32]), derivative_matrix(32)), [-1, 32, 1])
        MAX5 = tf.layers.max_pooling1d(inputs=MAX4, pool_size=2, strides=2, padding='same')
        MIN5 = -tf.layers.max_pooling1d(inputs=-MIN4, pool_size=2, strides=2, padding='same')
        F5 = tf.concat([M5,MAX5,AVG5,MIN5],axis=2)

        # Conv -> (?, 32, 32)
        L6 = tf.layers.conv1d(inputs=F5, filters=32, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu, name='C6', reuse=tf.AUTO_REUSE)
        M6 = tf.layers.max_pooling1d(inputs=L6, pool_size=2, strides=2, padding='same')
        AVG6 = tf.layers.average_pooling1d(inputs=AVG5, pool_size=2, strides=2, padding='same')
        #DRV6 = tf.reshape(tf.matmul(tf.reshape(AVG6, [-1, 16]), derivative_matrix(16)), [-1, 16, 1])
        MAX6 = tf.layers.max_pooling1d(inputs=MAX5, pool_size=2, strides=2, padding='same')
        MIN6 = -tf.layers.max_pooling1d(inputs=-MIN5, pool_size=2, strides=2, padding='same')
        F6 = tf.concat([M6,MAX6,AVG6,MIN6],axis=2)
        # Conv -> (?, 16, 32)

        L7 = tf.layers.conv1d(inputs=F6, filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='C7', reuse=tf.AUTO_REUSE)
        M7 = tf.layers.max_pooling1d(inputs=L7, pool_size=2, strides=2, padding='same')
        AVG7 = tf.layers.average_pooling1d(inputs=AVG6, pool_size=2, strides=2, padding='same')
        #DRV7 = tf.reshape(tf.matmul(tf.reshape(AVG7, [-1, 8]), derivative_matrix(8)), [-1, 8, 1])
        MAX7 = tf.layers.max_pooling1d(inputs=MAX6, pool_size=2, strides=2, padding='same')
        MIN7 = -tf.layers.max_pooling1d(inputs=-MIN6, pool_size=2, strides=2, padding='same')
        F7 = tf.concat([M7,MAX7,AVG7,MIN7],axis=2)
        # Conv -> (?, 8, 64)

        L8 = tf.layers.conv1d(inputs=F7, filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu, name='C8', reuse=tf.AUTO_REUSE)
        M8 = tf.layers.max_pooling1d(inputs=L8, pool_size=2, strides=2, padding='same')
        AVG8 = tf.layers.average_pooling1d(inputs=AVG7, pool_size=2, strides=2, padding='same')
#        DRV8 = tf.reshape(tf.matmul(tf.reshape(AVG8, [-1, 4]), derivative_matrix(4)), [-1, 4, 1])
        MAX8 = tf.layers.max_pooling1d(inputs=MAX7, pool_size=2, strides=2, padding='same')
        MIN8 = -tf.layers.max_pooling1d(inputs=-MIN7, pool_size=2, strides=2, padding='same')
#        F8 = tf.concat([M8,MAX8,AVG8,DRV8,MIN8],axis=2)
        F8 = tf.concat([M8,MAX8,AVG8,MIN8],axis=2)
        # Conv -> (?, 4, 64)

        flat = tf.reshape(F8, (-1, 256+4*3))
        Output = Lambda(lambda x :tf.add(tf.layers.dense(flat, 1, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE),0,name='Output'))

        model = Sequential()
        model.add(Lambda(lambda x :tf.add(tf.layers.dense(flat, 1, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), reuse=tf.AUTO_REUSE),0,name='Output'),input_shape=(2000,3)))


        return Output


def semi_sggnet_ver2():
    input_layer= Input(shape=(1000, 3))
    x = Conv1D(128, 12, activation='relu')(input_layer)
    x = Conv1D(128, 12, activation='relu')(x)
    x1 = keras.layers.AveragePooling1D(2, 2)(x)
    x2 = Conv1D(128, 12,strides=2, padding='SAME' , activation='relu')(x)
    x = keras.layers.concatenate([x1,x2])
    x = Conv1D(128, 12, activation='relu')(x)
    x = Conv1D(128, 12, activation='relu')(x)
    x1 = keras.layers.AveragePooling1D(2, 2,padding='same')(x)
    x2 = Conv1D(128, 12,strides=2, padding='same' , activation='relu')(x)
    x = keras.layers.concatenate([x1,x2])
    x = Conv1D(256, 8, activation='relu')(x)
    x = Conv1D(256, 8, activation='relu')(x)
    x1 = keras.layers.AveragePooling1D(2, 2,padding='same')(x)
    x2 = Conv1D(256, 8,strides=2, padding='SAME' , activation='relu')(x)
    x = keras.layers.concatenate([x1,x2])
    x = Conv1D(256, 8, activation='relu')(x)
    x = Conv1D(256, 8, activation='relu')(x)
    x1 = keras.layers.AveragePooling1D(2, 2,padding='same')(x)
    x2 = Conv1D(256, 8,strides=2, padding='SAME' , activation='relu')(x)
    x = keras.layers.concatenate([x1,x2])
    x = Conv1D(512, 5, activation='relu')(x)
    x = Conv1D(512, 5, activation='relu')(x)
    x1 = keras.layers.AveragePooling1D(2, 2,padding='same')(x)
    x2 = Conv1D(512, 5,strides=2, padding='SAME' , activation='relu')(x)
    x = keras.layers.concatenate([x1,x2])
    x = Conv1D(512, 3, activation='relu')(x)
    x = Conv1D(512, 3, activation='relu')(x)
    x1 = keras.layers.AveragePooling1D(2, 2,padding='same')(x)
    x2 = Conv1D(512, 3,strides=2, padding='SAME' , activation='relu')(x)
    x = keras.layers.concatenate([x1,x2])
    x = Conv1D(1024, 3, activation='relu')(x)
    x = Conv1D(1024, 3, activation='relu')(x)
    x1 = keras.layers.AveragePooling1D(2, 2,padding='same')(x)
    x2 = Conv1D(1024, 3,strides=2, padding='SAME' , activation='relu')(x)

    x = keras.layers.concatenate([x1,x2])
    x = Conv1D(1024, 2, activation='relu')(x)
    #x = Conv1D(1024, 3, activation='relu')(x)
    x = keras.layers.SpatialDropout1D(0.05)(x)
    # model.add(layers.Dense(512, activation='relu'))
    x = keras.layers.Flatten()(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    output_layer = x

    model = Model(input_layer, output_layer)

    return model


def semi_vggnet1000_maxpool():

    # Design model
    model = Sequential()
    model.add(Conv1D(64, 12, activation='relu',
                     input_shape=(1000, 3)))
    model.add(Conv1D(64, 12, activation='relu'))
    model.add(keras.layers.MaxPooling1D(2, 2))
    model.add(Conv1D(128, 12, activation='relu'))
    model.add(Conv1D(128, 12, activation='relu'))
    model.add(keras.layers.MaxPooling1D(2, 2))
    model.add(Conv1D(128, 8, activation='relu'))
    model.add(Conv1D(128, 8, activation='relu'))
    model.add(keras.layers.MaxPooling1D(2, 2))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(keras.layers.MaxPooling1D(2, 2))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(Conv1D(256, 8, activation='relu'))
    model.add(keras.layers.MaxPooling1D(2, 2))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(keras.layers.MaxPooling1D(2, 2))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(keras.layers.MaxPooling1D(2, 2))
    model.add(keras.layers.Dropout(0.05))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))

    return model


def semi_vggnet():
    # Design model
    model = Sequential()
    model.add(Conv1D(64, 12, activation='relu',
                            input_shape=(2000,3)))
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
    model.add(keras.layers.Dropout(0.5))
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
    model.add(keras.layers.Dropout(0.05))
    # model.add(layers.Dense(512, activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(Dense(1, activation='linear'))

    return model

def semi_vggnet1000_stride():
    # Design model
    model = Sequential()
    model.add(Conv1D(64, 12, activation='relu',
                            input_shape=(1000,3)))
    model.add(Conv1D(64, 12,strides=2, activation='relu'))
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
    model.add(Conv1D(1024, 3, activation='relu'))
    model.add(keras.layers.Dropout(0.05))
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

