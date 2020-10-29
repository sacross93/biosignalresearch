from utils.my_classes import DataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from utils.processing import *
from utils.models import *
import csv
from utils.resnet1D import resnet1D_model
import keras
from SVV_projects.read_SVV import read_abp_svv, read_abp_svv_10sec,read_abp_svv_minmax_fft

from keras.utils.training_utils import multi_gpu_model

import sklearn.metrics
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D,Dense,LSTM,Dropout
import pymysql
import os
from tcn.tcn import  compiled_tcn
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3";



# model = resnet152_model()
# model.summary()

#files = [ '180206', '180207', '180208', '180209',  '180213', '180214']
savepath = '/home/projects/pcg_transform/pcg_AI/deep/SVV/result/'+'vgglike_5month_no_arrhy/'

if not os.path.isdir(savepath ):
    os.mkdir(savepath )



stdate = 180201
enddate = 180631

valstdate = 180701
valenddate = 181201


conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                       db='abp_svv_generator', charset='utf8')
curs = conn.cursor()

sql = """select file_name,EV_SVV from abp_svv_generator.svv_test1 where date >=%s and date <=%s order by date,file_name;"""
curs.execute(sql,(stdate,enddate))
row = curs.fetchall()

training_data = np.array(row)[:,0]
labels = {row[i][0]: row[i][1] for i in range(len(row))}



sql = """select file_name,EV_SVV from abp_svv_generator.svv_test1 where date >=%s and date <=%s order by date,file_name;"""
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




""""""
filedate = str(stdate)

""""""



params = {'dim': (1000,),
          'batch_size': 32,
          'n_classes': 1,
          'n_channels': 3,
          'shuffle': False}

val_params = {'dim': (1000,),
              'batch_size': 32,
              'n_classes': 1,
              'n_channels': 3,
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



model = semi_vggnet1000_stride()

model = multi_gpu_model(model, gpus=2)
[...]  # Architecture
adam = optimizers.Adam(lr=0.00001, clipnorm=1.)
model.compile(adam, loss='mean_squared_error')

# Train model on dataset
#model.load_weights('/home/jmkim/PycharmProjects/keras_deep_pp/tmp.h5')


bestpath = savepath + '5month_abp_ssv.h5'
#model.load_weights(bestpath)

#model. load_weights(bestpath)
epoch = 20

best = 99999
comple_flag = 0

val_mse = []
cnt = 0


model. load_weights(bestpath)


plt.plot(val_mse, 'b', label='Validation mean_square_error')
plt.title(filedate + 'vgglike Training and validation mean_square_error')
plt.legend()
# plt.show()
#plt.savefig(
#    '/home/projects/pcg_transform/pcg_AI/deep/SVV/result/self_graph_tcn01/' + filedate + 'loss_val_SVV_self_128_8_t2' + '.png')
plt.show()
plt.close()

val_params = {'dim': (1000,),
              'batch_size': 1,
              'n_classes': 1,
              'n_channels': 3,
              'shuffle': False}


savepath = '/home/projects/pcg_transform/pcg_AI/deep/SVV/result/'+'vgglike_arrytest/'


if not os.path.isdir(savepath ):
    os.mkdir(savepath )
validation_files=range(190101,190301)



for i in range(len(validation_files)):

    val_file = validation_files[i]
    print('start : ', val_file)

    conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                           db='abp_svv_generator', charset='utf8')
    curs = conn.cursor()


    sql = """select distinct room_name from abp_svv_generator.abp_svv where date =%s;"""
    curs.execute(sql, (val_file))
    rooms = curs.fetchall()

    conn.close()
    print(rooms)



    if len(rooms)<1:
        continue

    for room in rooms:
        room = list(room)[0]
        conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                               db='abp_svv_generator', charset='utf8')
        curs = conn.cursor()

        sql = """select file_name,EV_SVV,room_name,date from abp_svv_generator.abp_svv_dataset where date =%s and room_name = %s;"""
        curs.execute(sql, (val_file,room))
        row = curs.fetchall()
        conn.close()

        if len(row) == 0:
            print('have not database')
            continue





        val_data = np.array(row)[:, 0]
        val_y = np.array(row)[:, 1]
        #room_name = np.array(row)[:, 2]
        date = np.array(row)[:, 3]
        val_labels = {row[i][0]: row[i][1] for i in range(len(row))}

        labels =val_labels

        if len(val_data) < 1000:
            continue


        partition2 = {'val': val_data}

        validation_generator2 = DataGenerator(partition2['val'], labels, **val_params, path='')

        pred = model.predict_generator(generator=validation_generator2)
        #result = model.evaluate_generator(generator=validation_generator2)
        tmppred = pred[:]

        import scipy.signal

        pred = scipy.signal.savgol_filter(tmppred.flatten(),449,0)
        val_y = val_y.flatten()
        val_y_filtered = scipy.signal.savgol_filter(val_y,449,0)

        mae = sklearn.metrics.mean_absolute_error(val_y_filtered.astype('float32'),pred)

        corr = np.corrcoef(pred, val_y_filtered.astype('float32'))[0, 1]

        fig = plt.figure(figsize=(20, 10))
        plt.title(room + "_" +str(val_file) + "_Mean_absolued_error :" + str(mae)[:5] + "  corr :" + str(corr)[:4])
        plt.plot(pred, 'b', label="Predicted")
        plt.plot(val_y_filtered, 'r', label='SVV')
        if float(max(val_y)) > 12.0:
            plt.axhline(12, color='black', ls='--', linewidth=1)
        #plt.xticks([i * (len(pred) // 5 - 1) for i in range(6)])
        plt.legend(fontsize='xx-large')

        plt.savefig(
           savepath + room + "_" + str(val_file) + 'SVV_no_arri' + '.png')
        #plt.show()
        plt.close()


        #for j in
        #array_bae_mean[output_cnt],array_bae_diff[output_cnt] = bland_altman_result(pred.flatten()[1::20], val_y_filtered.astype('float32')[1::20])
        #output_cnt = output_cnt+1
        #bland_altman_plot_20seg(pred.flatten(), val_y_filtered.astype('float32'))


        with open(savepath + 'result.csv', 'a',
                  newline='') as corr_csv:
            wr = csv.writer(corr_csv)
            wr.writerow([room,val_file,str(mae)[:5],str(corr)[:4]])
