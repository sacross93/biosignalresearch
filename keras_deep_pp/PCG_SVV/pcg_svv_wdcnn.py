from utils.my_classes import DataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from utils.processing import *
import csv
from utils.resnet1D import resnet1D_model
import keras
from SVV_projects.read_SVV import read_abp_svv, read_abp_svv_10sec,read_abp_svv_minmax_fft
from PCG_SVV.pcg_models import *
from keras.utils.training_utils import multi_gpu_model

import sklearn.metrics
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D,Dense,LSTM,Dropout
import pymysql
import os
from tcn.tcn import  compiled_tcn
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "3";



folder_name = 'model_wcnn_fn/'

savepath = '/home/projects/pcg_transform/pcg_AI/deep/PCG_SVV/result/'+folder_name

if not os.path.isdir(savepath ):
    os.mkdir(savepath )

# model = resnet152_model()
# model.summary()

#files = [ '180206', '180207', '180208', '180209',  '180213', '180214']


conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                       db='abp_svv_generator', charset='utf8')
curs = conn.cursor()

sql = """select file_name,EV_SVV from abp_svv_generator.pcg_ev1000 order by date,file_name;"""
curs.execute(sql)
row = curs.fetchall()



training_data = np.array(row)[:,0]
labels = {row[i][0]: row[i][1] for i in range(len(row))}


conn.close()


val_data = training_data[len(training_data)//2:]
val_y = np.array(row)[:,1][len(training_data)//2:]
training_data = training_data[:len(training_data)//2]

"""


files = ['180301','180303','180304','180305','180307']
training_files =files[:3]
validation_files=[files[3]]
test_files=[files[4]]
"""


""""""
filedate = str(190221)

""""""



params = {'dim': (10000,),
          'batch_size': 64,
          'n_classes': 1,
          'n_channels': 5,
          'shuffle': False}

val_params = {'dim': (10000,),
              'batch_size': 64,
              'n_classes': 1,
              'n_channels': 5,
              'shuffle': False}



#data, labels, y_time, y_data = get_pcg_svv_data([filedate], x_path=datapath,y='SVV')



partition = {'train': training_data, 'validation': val_data}
# labels = {'180313_0_1' : 57.08444, '180313_0_2' :75.627174, '180313_0_0' : 73.48143 }
# labels = {'180313_0_1' : 1, '180313_0_2' :0, '180313_0_0' : 1 }
# [57.08444 , 75.627174, 73.48143]


# Generators
training_generator = DataGenerator(partition['train'], labels, **params, path='')
validation_generator = DataGenerator(partition['validation'], labels, **val_params, path='')

#if len(x_data == 0:
#   continue
import keras.backend.tensorflow_backend as K
#with K.tf.device('/gpu:3'):



model = widecnn_pcg()

#model = multi_gpu_model(model, gpus=2)
[...]  # Architecture
adam = optimizers.Adam(lr=0.00005, clipnorm=1.)
model.compile(adam, loss='mean_squared_error')

# Train model on dataset
#model.load_weights('/home/jmkim/PycharmProjects/keras_deep_pp/tmp.h5')


bestpath = '/home/projects/pcg_transform/pcg_AI/deep/SVV/saved_weight/' + filedate + '_self_pcg_to_svv_test01.h5'
#model.load_weights(bestpath)

#model. load_weights(bestpath)
epoch = 20

best = 99999
comple_flag = 0

val_mse = []
cnt = 0

for i in range(epoch):
    print('epoch : ' + str(i))
    history = model.fit_generator(generator=training_generator, epochs=1, validation_data=validation_generator )
    mse = history.history['val_loss'][0]
    val_mse.append(mse)
    print(mse)
    if mse < best:
        print('saved_best_data')
        best = mse
        model.save_weights(
            bestpath)
        cnt = 0
    else: cnt = cnt+1
    #if cnt >=10:
    #    break

#model.save_weights('tmp_all_march_stride.h5')
#model.load_weights('tmp.h5')

model. load_weights(bestpath)
# model.load_weights(bestpath + filedate + 'best_resnet152_predict_SV_test.h5')
#model.save_weights('tmp_all_3month_vgglike_stride_ver2.h5')
#model.save_weights('tmp_all_3month_vgglike_stride2.h5')
#model.load_weights('tmp_all_3month_vgglike_stride2.h5')

plt.plot(val_mse, 'b', label='Validation mean_square_error')
plt.title(filedate + 'vgglike Training and validation mean_square_error')
plt.legend()
# plt.show()
#plt.savefig(
#    '/home/projects/pcg_transform/pcg_AI/deep/SVV/result/self_graph_tcn01/' + filedate + 'loss_val_SVV_self_128_8_t2' + '.png')
plt.show()
plt.close()




val_params = {'dim': (10000,),
              'batch_size': 1,
              'n_classes': 1,
              'n_channels': 5,
              'shuffle': False}
validation_generator = DataGenerator(partition['validation'], labels, **val_params, path='')

pred = model.predict_generator(generator=validation_generator)
result = model.evaluate_generator(generator=validation_generator)
tmppred = pred[:]

import scipy.signal

pred = scipy.signal.savgol_filter(tmppred.flatten(),449,0)
val_y = val_y.flatten()
val_y_filtered = scipy.signal.savgol_filter(val_y,449,0)

mae = sklearn.metrics.mean_absolute_error(val_y_filtered.astype('float32'),pred)

corr = np.corrcoef(pred.flatten(), val_y_filtered.astype('float32'))[0, 1]

fig = plt.figure(figsize=(20, 10))
plt.title("_Mean_absolued_error :" + str(mae)[:5] + "  corr :" + str(corr)[:4])
plt.plot(pred, 'b', label="Predicted")
plt.plot(val_y_filtered, 'r', label='SVV')
#if np.max(val_y_filtered) > 12:
#    plt.axhline(12, color='black', ls='--', linewidth=1)
#plt.xticks([i * (len(pred) // 5 - 1) for i in range(6)])
plt.legend(fontsize='xx-large')
plt.show()
plt.savefig(
    '/home/projects/pcg_transform/pcg_AI/deep/SVV/pcg_result/self_result/pcg_svv_wide02' + '.png')
#plt.show()
#plt.close()


