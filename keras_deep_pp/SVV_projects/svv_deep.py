from utils.my_classes import DataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from utils.processing import get_ml_pp_data
from utils.models import semi_vggnet,semi_vggnet1000
import csv
from utils.resnet1D import resnet1D_model
import keras
from SVV_projects.read_SVV import read_abp_svv, read_abp_svv_10sec,read_abp_svv_minmax_fft
from keras.utils.training_utils import multi_gpu_model

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D,Dense,LSTM,Dropout

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3";



# model = resnet152_model()
# model.summary()

#files = [ '180206', '180207', '180208', '180209',  '180213', '180214']


training_files = []

for i in range(20):
    tmp = 180301+i
    training_files.append(str(tmp))


validation_files=[]
for i in range(6):
    tmp = 180401+i
    validation_files.append(str(tmp))





training_files = ['180303','180304','180311','180316','180317','180320','180323']

"""


files = ['180301','180303','180304','180305','180307']
training_files =files[:3]
validation_files=[files[3]]
test_files=[files[4]]
"""

save_path = '/home/projects/pcg_transform/pcg_AI/deep/SVV/result/graph/'



filedate = training_files[0]

""""""
x_data, y_data = read_abp_svv(training_files)
training_data =x_data
ty = y_data

# validation
val_filedate = validation_files[0]

#x_data, y_data = read_abp_svv(files[4:])
x_data, y_data = read_abp_svv(validation_files)
validation_data = x_data
val_y = y_data

#if len(x_data == 0:
#   continue
import keras.backend.tensorflow_backend as K
#with K.tf.device('/gpu:3'):


input = keras.layers.Input(shape=(2000, 3), name='data')
# Design model
model = semi_vggnet1000(input)



[...]  # Architecture
opt = optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model = multi_gpu_model(model, gpus=2)

model.compile(loss='mse', optimizer=opt, metrics=['mse'])
# Train model on dataset
#model.load_weights('/home/jmkim/PycharmProjects/keras_deep_pp/tmp.h5')

bestpath = '/home/projects/pcg_transform/pcg_AI/deep/SVV/saved_weight/' + filedate + 'best_resnet_predict_SVV_all.h5'

epoch = 10

best = 99999
comple_flag = 0
save_flag = 0
val_mse = []

for j in range(10):
    if comple_flag == 1:
        break
    if j > 1:
        save_flag = 1
    for i in range(epoch):
        print('epoch : ' + str((i + 1) + (j) * 10))
        history = model.fit(training_data, ty, validation_data=(validation_data, val_y))
        mse = history.history['val_mean_squared_error'][0]
        val_mse.append(mse)
        print(mse)
        if mse < best and save_flag == 1:
            print('saved_best_data')
            best = mse
            model.save_weights(
                bestpath)

        if mse > best * 5:
            print('Training_complete')
            comple_flag = 1
            break

model.save_weights('tmp.h5')
#model.load_weights('tmp.h5')

model.load_weights(bestpath)
# model.load_weights(bestpath + filedate + 'best_resnet152_predict_SV_test.h5')

# model.save_weights('/home/projects/pcg_transform/pcg_AI/deep/SV/save_weight/'+filedate+'resnet152_predict_SV_test.h5')

#mse = history.history['mean_squared_error']
#val_mse = history.history['val_mean_squared_error']

#epochs = range(len(mse))

plt.close(1)  # To clear previous figure
plt.close(2)

# plt.ylim(0,300)
# plt.plot( 'bo', label='Training mean_square_error')
plt.plot(val_mse, 'b', label='Validation mean_square_error')
plt.title(filedate + 'Training and validation mean_square_error')
plt.legend()
#plt.show()
plt.savefig(
    '/home/projects/pcg_transform/pcg_AI/deep/SVV/result/graph_test03/' + filedate + '_' + val_filedate + 'loss_val_SVV_' + '.png')
plt.show()
plt.close()
# plt.figure()



for i in range(len(validation_files)):

    #test_files.append(str(tmp))
    val_file = validation_files[i]

    val_filedate = validation_files[i]
    x_data, y_data = read_abp_svv([val_file])
    validation_data = x_data
    val_y = y_data

    if len(x_data)==0:
        continue


    pred = model.predict(validation_data)
    result = model.evaluate(validation_data, val_y)
    tmppred = pred[:]

    import scipy.signal

    pred = scipy.signal.savgol_filter(tmppred.flatten(),149,0)
    val_y = val_y.flatten()
    val_y_filtered = scipy.signal.savgol_filter(val_y,149,0)

    corr = np.corrcoef(pred, val_y_filtered.astype('float32'))[0, 1]
    plt.title(str('val') + val_filedate + "_Mean_square_error :" + str(result[1])[:4] + "  corr :" + str(corr)[:4])
    plt.plot(pred, 'b', label="Predicted")
    plt.plot(val_y_filtered, 'r', label='SVV')
    plt.xticks([i * (len(pred) // 5 - 1) for i in range(6)])
    plt.legend()
    plt.savefig(
        '/home/projects/pcg_transform/pcg_AI/deep/SVV/result/graph_test03/' + filedate + '_' + val_filedate + 'val_SVV' + '.png')
    plt.show()
    plt.close()

