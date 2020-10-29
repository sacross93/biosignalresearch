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
import pymysql
import os
from tcn.tcn import  compiled_tcn
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "2";



# model = resnet152_model()
# model.summary()

#files = [ '180206', '180207', '180208', '180209',  '180213', '180214']




stdate = 180201
enddate = 180229

valstdate = 180302
valenddate = 180311




conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                       db='abp_svv_generator', charset='utf8')
curs = conn.cursor()

sql = """select file_name,EV_SVV from abp_svv_generator.abp_svv where date >=%s and date <=%s order by date,file_name;"""
curs.execute(sql,(stdate,enddate))
row = curs.fetchall()

training_data = np.array(row)[:,0]
labels = {row[i][0]: row[i][1] for i in range(len(row))}



sql = """select file_name,EV_SVV from abp_svv_generator.abp_svv where date >=%s and date <=%s order by date,file_name;"""
curs.execute(sql,(valstdate,valenddate))
row = curs.fetchall()

val_data = np.array(row)[:,0]
val_y = np.array(row)[:,1]
val_labels = {row[i][0]: row[i][1] for i in range(len(row))}

conn.close()

labels.update(val_labels)

"""


files = ['180301','180303','180304','180305','180307']
training_files =files[:3]
validation_files=[files[3]]
test_files=[files[4]]
"""

save_path = '/home/projects/pcg_transform/pcg_AI/deep/SVV/result/graph/'


""""""
filedate = str(stdate)

""""""



params = {'dim': (1000,),
          'batch_size': 64,
          'n_classes': 1,
          'n_channels': 3,
          'shuffle': False}

val_params = {'dim': (1000,),
              'batch_size': 1,
              'n_classes': 1,
              'n_channels': 3,
              'shuffle': True}



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


model = compiled_tcn(return_sequences=False,
                     num_feat=3,
                     num_classes=0,
                     nb_filters=128,
                     kernel_size=8,
                     dilations=[2 ** i for i in range(9)],
                     nb_stacks=2,
                     max_len=1000,
                     activation='norm_relu',
                     use_skip_connections=True,
                     regression=True,
                     dropout_rate=0.05,
                     modelflag=1)


#model = multi_gpu_model(model, gpus=3)
[...]  # Architecture
adam = optimizers.Adam(lr=0.00001, clipnorm=1.)
model.compile(adam, loss='mean_squared_error')

# Train model on dataset
#model.load_weights('/home/jmkim/PycharmProjects/keras_deep_pp/tmp.h5')

bestpath = '/home/projects/pcg_transform/pcg_AI/deep/SVV/saved_weight/' + filedate + 'best_resnet_predict_SVV_all.h5'
#model.load_weights(bestpath)


epoch = 100

best = 99999
comple_flag = 0
bestepoch = -1
val_mse = []


for i in range(epoch):
    print('epoch : ' + str(i ))
    history = model.fit_generator(generator=training_generator, epochs=1, validation_data=validation_generator,validation_steps=3)
    mse = history.history['val_loss'][0]
    val_mse.append(mse)
    print(mse)
    if mse < best:
        print('saved_best_data')
        best = mse
        bestepoch = epoch
        model.save_weights(
            bestpath)

model.save_weights('tmp.h5')
#model.load_weights('tmp.h5')

model.load_weights(bestpath)
# model.load_weights(bestpath + filedate + 'best_resnet152_predict_SV_test.h5')
model.save_weights('tcn_v2_128_8_march.h5')


plt.plot(val_mse, 'b', label='Validation mean_square_error')
plt.title(filedate + 'vgglike t1 Training and validation mean_square_error')
plt.legend()
# plt.show()
#plt.savefig(
#    '/home/projects/pcg_transform/pcg_AI/deep/SVV/result/self_graph_tcn01/' + filedate + 'loss_val_SVV_self_128_8_t2' + '.png')
plt.show()
#plt.close()


validation_files=[]
for i in range(30):
    tmp = 180301+i
    validation_files.append(str(tmp))


for i in range(len(validation_files)):

    #test_files.append(str(tmp))
    val_file = validation_files[i]

    val_filedate = validation_files[i]
    x_data, y_data = read_abp_svv_10sec([val_file])
    validation_data_test = x_data
    val_y = y_data

    if len(x_data)==0:
        continue


    pred = model.predict(validation_data_test)
    result = model.evaluate(validation_data_test, val_y)
    tmppred = pred[:]

    import scipy.signal

    pred = scipy.signal.savgol_filter(tmppred.flatten(),449,0)
    val_y = val_y.flatten()
    val_y_filtered = scipy.signal.savgol_filter(val_y,449,0)

    corr = np.corrcoef(pred, val_y_filtered.astype('float32'))[0, 1]
    fig = plt.figure(figsize=(20, 10))
    plt.title(val_filedate + "_Mean_square_error :" + str(result)[:4] + " corr :" + str(corr)[:4])
    plt.plot(pred, 'b', label="Predicted")
    plt.plot(val_y_filtered, 'r', label='SVV')
    plt.xticks([i * (len(pred) // 5 - 1) for i in range(6)])
    plt.legend()
    plt.savefig(
        '/home/projects/pcg_transform/pcg_AI/deep/SVV/result/graph_tcn03/' + filedate + '_' + val_filedate + 'val_march' + '.png')
    #plt.show()
    plt.close()

