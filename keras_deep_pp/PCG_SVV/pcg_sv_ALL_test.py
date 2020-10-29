import sklearn
import sklearn.metrics
from utils.my_classes import DataGenerator_npz
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from utils.processing import *
from utils.models import *
import csv
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
import datetime
# The GPU id to use, usually either "0" or "1";
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3";
from keras.utils.training_utils import multi_gpu_model
import keras
#from SVV_projects.read_SVV import read_abp_svv, read_abp_svv_10sec,read_abp_svv_minmax_fft
from PCG_SVV.pcg_models import *

#import sklearn.metrics
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D,Dense,LSTM,Dropout
import pymysql
import os
from tcn.tcn import  compiled_tcn
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,3";

type = 'minmax'
folder_name = 'pcg_sv_004_2018_2019_training/'

savepath = '/home/projects/pcg_transform/pcg_AI/deep/PCG_SVV/result/'+folder_name
if not os.path.isdir(savepath ):
    os.mkdir(savepath )
# model = resnet152_model()
# model.summary()





#files = [ '180206', '180207', '180208', '180209',  '180213', '180214']

HOSTNAME = 'localhost'
USERNAME = 'root'
PASSWORD = 'signal@anes'
DBNAME = 'abp_svv_generator'
DEVICE_DB_NAME = 'Vital_DB'


stdate = 180101
enddate = 190831

# stdate2 = 190801
# enddate2 = 191231

valstdate = 190901
valenddate = 191231

room_name = 'D-06'

conn = pymysql.connect(host='localhost', user=USERNAME, password=PASSWORD,
                       db=DBNAME, charset='utf8')
curs = conn.cursor()

sql = """select pcg_file,SV from PCG_SV where    date >=%s and date <=%s and room_name = %s  and type = %s   order by date,pcg_file;"""
curs.execute(sql,(stdate,enddate,room_name,type))
row = curs.fetchall()

training_data = np.array(row)[:,0]
labels = {row[i][0]: np.array(row[i][1],dtype=np.float16) for i in range(len(row))}
#
# sql = """select pcg_file,SVV from data_generator.PCG_SVV where date >=%s and date <=%s and type = %s and room_name = 'D-02' order by room_name,date,pcg_file;"""
# curs.execute(sql,(stdate2,enddate2,type))
# row = curs.fetchall()
#
# training_data = np.concatenate([training_data,np.array(row)[:,0]])
#
# t2_labels = {row[i][0]: np.array(row[i][1],dtype=np.float16) for i in range(len(row))}
# labels.update(t2_labels)


sql = """select pcg_file,SV from PCG_SV where  date >=%s and date <=%s  and room_name = %s and type = %s order  by room_name, date,pcg_file;"""
curs.execute(sql,(valstdate,valenddate,room_name,type))
row = curs.fetchall()

val_data = np.array(row)[:,0]
val_y = np.array(row)[:,1]
val_y = np.array(val_y,dtype=np.float16)

val_labels = {row[i][0]: np.array(row[i][1],dtype=np.float16) for i in range(len(row))}

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



params = {'dim': (3000,),
          'batch_size': 16,
          'n_classes': 1,
          'n_channels': 2,
          'shuffle': False}

val_params = {'dim': (3000,),
              'batch_size': 16,
              'n_classes': 1,
              'n_channels': 2,
              'shuffle': False}



#data, labels, y_time, y_data = get_pcg_svv_data([filedate], x_path=datapath,y='SVV')



partition = {'train': training_data, 'validation': val_data}
# labels = {'180313_0_1' : 57.08444, '180313_0_2' :75.627174, '180313_0_0' : 73.48143 }
# labels = {'180313_0_1' : 1, '180313_0_2' :0, '180313_0_0' : 1 }
# [57.08444 , 75.627174, 73.48143]


# Generators
training_generator = DataGenerator_npz(partition['train'], labels, **params, path='')
validation_generator = DataGenerator_npz(partition['validation'], labels, **val_params, path='')

#if len(x_data == 0:
#   continue
import keras.backend.tensorflow_backend as K
#with K.tf.device('/gpu:3'):



model = pcg_SV_testmodel01()

model = multi_gpu_model(model, gpus=4)
[...]  # Architecture
adam = optimizers.Adam(lr=0.00001, clipnorm=1.)
model.compile(adam, loss='mean_squared_error')

# Train model on dataset
#model.load_weights('/home/jmkim/PycharmProjects/keras_deep_pp/tmp.h5')


bestpath = savepath + filedate + 'pcg_sv_20191104.h5'
current_path = savepath + filedate + 'pcg_sv_last.h5'
#model.load_weights(bestpath)

#model. load_weights(bestpath)
epoch = 100

best = 99999
comple_flag = 0

val_mse = []
loss_mse = []
cnt = 0
#model. load_weights(current_path)

for i in range(epoch):
    print('epoch : ' + str(i))
    history = model.fit_generator(generator=training_generator, epochs=1, validation_data=validation_generator )
    mse = history.history['val_loss'][0]
    loss_mse.append(history.history['loss'][0])
    val_mse.append(mse)
    print(mse)

    plt.plot(loss_mse, 'green', label='Training mean_square_error')
    plt.plot(val_mse, 'b', label='Validation mean_square_error')

    plt.title(filedate + 'vgglike Training and validation mean_square_error')
    plt.legend()
    # plt.show()
    plt.savefig(
        savepath + 'loss_val_SV' + '.png')
    plt.close()

    if i ==0:
        continue

    if mse < best:
        print('saved_best_data')
        best = mse
        model.save_weights(
            bestpath)
        cnt = 0

    else: cnt = cnt+1
    if cnt >=10:
        break

model.save_weights(current_path)
#model.save_weights('tmp_all_march_stride.h5')
#model.load_weights('tmp.h5')

model. load_weights(bestpath)
#model. load_weights(current_path)
#model. load_weights(bestpath)
# model.load_weights(bestpath + filedate + 'best_resnet152_predict_SV_test.h5')
#model.save_weights('tmp_all_3month_vgglike_stride_ver2.h5')
#model.save_weights('tmp_all_3month_vgglike_stride2.h5')
#model.load_weights('tmp_all_3month_vgglike_stride2.h5')

validation_files=[]
for i in range(131):
    tmp = 190901+i
    validation_files.append(str(tmp))

val_params = {'dim': (3000,),
              'batch_size': 1,
              'n_classes': 1,
              'n_channels': 2,
              'shuffle': False}

output_cnt = 0
for i in range(len(validation_files)):

    val_file = validation_files[i]
    print('start : ', val_file)

    conn = pymysql.connect(host='localhost', user=USERNAME, password=PASSWORD,
                           db=DBNAME, charset='utf8')
    curs = conn.cursor()




    sql = """select distinct room_name from PCG_SV where date =%s  and room_name = %s   order by room_name;"""
    curs.execute(sql, (val_file,room_name))
    rooms = curs.fetchall()

    conn.close()
    print(rooms)



    if len(rooms)<1:
        continue

    for room in rooms:
        room = list(room)[0]
        conn = pymysql.connect(host='localhost', user=USERNAME, password=PASSWORD,
                               db=DBNAME, charset='utf8')
        curs = conn.cursor()

        sql = """select pcg_file,SV,time from PCG_SV where date =%s and room_name =%s and type = %s order by room_name,date,pcg_file;"""
        curs.execute(sql, (val_file,room,type))
        row = curs.fetchall()
        conn.close()

        if len(row) == 0:
            print('have not database')
            continue


        val_data = np.array(row)[:, 0]
        val_y = np.array(row)[:, 1]
        val_y = np.array(val_y,dtype=np.float16)
        #room_name = np.array(row)[:, 2]
        val_labels = {row[i][0]: val_y[i] for i in range(len(row))}
        val_time = np.array(row)[:, 2]

        labels.update(val_labels)

        if len(val_data) < 1800:
            continue


        partition2 = {'train': training_data, 'val': val_data}

        validation_generator2 = DataGenerator_npz(partition2['val'], labels, **val_params, path='')

        pred = model.predict_generator(generator=validation_generator2)
        #result = model.evaluate_generator(generator=validation_generator2)
        tmppred = pred[:]

        import scipy.signal

        val_time = np.array(val_time,dtype=np.datetime64)
        pred = scipy.signal.savgol_filter(tmppred.flatten(),449,0)
        val_y = val_y.flatten()
        val_y_filtered = scipy.signal.savgol_filter(val_y,449,0)

        mae = sklearn.metrics.mean_absolute_error(val_y_filtered.astype('float32'),pred)
        mse = sklearn.metrics.mean_squared_error(val_y_filtered.astype('float32'),pred)

        corr = np.corrcoef(pred, val_y_filtered.astype('float32'))[0, 1]

        fig = plt.figure(figsize=(20, 10))
        plt.title(room + "_" +val_file + "_Mean_absolued_error :" + str(mae)[:5] + "  corr :" + str(corr)[:4])
        plt.plot(val_time,pred, 'b', label="Predicted")
        plt.plot(val_time,val_y_filtered, 'r', label='SV')
        if float(max(val_y)) > 12.0:
            plt.axhline(12, color='black', ls='--', linewidth=1)
        #plt.xticks([i * (len(pred) // 5 - 1) for i in range(6)])
        plt.legend(fontsize='xx-large')

        plt.savefig(
           savepath + room + "_" + val_file + 'SV_no_arri' + '.png')
        #plt.show()
        plt.close()
