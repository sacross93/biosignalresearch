import sys
# sys.path.insert(0,'/home/jmkim/PycharmProjects/keras_deep_pp/')
# import os
# os.environ['LD_LIBRARY_PATH'] = os.getcwd()  # or whatever path you want
from utils.my_classes import DataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from utils.processing import *
from utils.models import *
import csv
#from utils.resnet1D import resnet1D_model
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3";

type = '2ch_dataset_sv'

folder_name = '2ch_dataset_sv_002/'
chnnal = 2

savepath = '/home/projects/pcg_transform/pcg_AI/deep/SVV/result/'+folder_name

if not os.path.isdir(savepath ):
    os.mkdir(savepath )



stdate = 180201
enddate = 180632

valstdate = 180701
valenddate = 181232

conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                       db='abp_svv_generator2', charset='utf8')
curs = conn.cursor()

sql = """select file_name,EV_SV from abp_sv_android where date >=%s and date <=%s and type = %s order by date,file_name;"""
curs.execute(sql,(stdate,enddate,type))
row = curs.fetchall()

training_data = np.array(row)[:,0]
labels = {row[i][0]: row[i][1] for i in range(len(row))}

#
# fig = plt.figure(figsize=(20, 10))
# plt.plot(np.arange(-12.2,12.2,0.0244),data[:,1])
# plt.savefig('/home/jmkim/abp_fft.png',dpi=300)
#


sql = """select file_name,EV_SV from abp_sv_android where date >=%s and date <=%s and type=%s order by date,file_name;"""
curs.execute(sql,(valstdate,valenddate , type))
row = curs.fetchall()

val_data = np.array(row)[:,0]
val_y = np.array(row)[:,1]
val_labels = {row[i][0]: row[i][1] for i in range(len(row))}

conn.close()

labels.update(val_labels)



""""""
filedate = str(stdate)

""""""



params = {'dim': (1000,),
          'batch_size': 64,
          'n_classes': 1,
          'n_channels': chnnal,
          'shuffle': False}

val_params = {'dim': (1000,),
              'batch_size': 64,
              'n_classes': 1,
              'n_channels': chnnal,
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



model = semi_vggnet1000_stride2c()

model = multi_gpu_model(model, gpus=4)
[...]  # Architecture
adam = optimizers.Adam(lr=0.000005, clipnorm=1.)
model.compile(adam, loss='mean_squared_error')

# Train model on dataset
#model.load_weights('/home/jmkim/PycharmProjects/keras_deep_pp/tmp.h5')


bestpath = savepath + filedate + '.h5'
#model.load_weights(bestpath)
currentpath = savepath + filedate + 'current.h5'
#model. load_weights(bestpath)
epoch = 100

best = 99999
comple_flag = 0

val_mse = []
cnt = 0
loss = []

for i in range(epoch):
    print('epoch : ' + str(i))
    history = model.fit_generator(generator=training_generator, epochs=1, validation_data=validation_generator )
    mse = history.history['val_loss'][0]
    loss_mse = history.history['loss'][0]
    loss.append(loss_mse)
    val_mse.append(mse)
    print(mse)

    plt.plot(loss, 'green', label='Training mean_square_error')
    plt.plot(val_mse, 'b', label='Validation mean_square_error')

    plt.title(filedate + 'vgglike Training and validation mean_square_error')
    plt.legend()
    # plt.show()
    plt.savefig(
        savepath + 'loss_val_SV' + '.png')
    plt.close()
    if mse < best:
        print('saved_best_data')
        best = mse
        model.save_weights(
            bestpath)
        cnt = 0
    else: cnt = cnt+1
    if cnt >=10:
        break

model.save_weights(currentpath)
# model.save(currentpath)
# model = model.get_layer('sequential_1')

#model.save_weights('tmp_all_march_stride.h5')
#model.load_weights('tmp.h5')

model.load_weights(bestpath)
# model.load_weights(bestpath + filedate + 'best_resnet152_predict_SV_test.h5')
#model.save_weights('tmp_all_3month_vgglike_stride_ver2.h5')
#model.save_weights('tmp_all_3month_vgglike_stride2.h5')
#model.load_weights('tmp_all_3month_vgglike_stride2.h5')

val_params = {'dim': (1000,),
              'batch_size': 1,
              'n_classes': 1,
              'n_channels': chnnal,
              'shuffle': False}

validation_files=[]


for i in range(31):
    tmp = 190101+i
    validation_files.append(str(tmp))

for i in range(31):
    tmp = 190201+i
    validation_files.append(str(tmp))


with open(savepath + 'all_data.csv', 'w',
          newline='') as corr_csv:
    wr = csv.writer(corr_csv)
    wr.writerow(['room', 'file', 'pred_sv', 'compared_sv'])

with open(savepath + 'result.csv', 'w',
          newline='') as corr_csv:
    wr = csv.writer(corr_csv)
    wr.writerow(['room', 'file', 'mae', 'mse', 'corr'])


array_bae_mean = [[0] for i in range(len(validation_files)*3)]
array_bae_diff = [[0] for i in range(len(validation_files)*3)]
output_cnt = 0
for i in range(len(validation_files)):

    val_file = validation_files[i]
    print('start : ', val_file)

    conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                           db='abp_svv_generator2', charset='utf8')
    curs = conn.cursor()


    sql = """select distinct room_name from abp_sv_android where date =%s;"""
    curs.execute(sql, (val_file))
    rooms = curs.fetchall()
    curs.close()
    conn.close()
    print(rooms)



    if len(rooms)<1:
        continue

    for room in rooms:
        room = list(room)[0]
        conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                               db='abp_svv_generator2', charset='utf8')
        curs = conn.cursor()

        sql = """select file_name,EV_SV,room_name,date,time from abp_sv_android where date =%s and room_name = %s and type = %s;"""
        curs.execute(sql, (val_file,room,type))
        row = curs.fetchall()
        curs.close()
        conn.close()

        if len(row) == 0:
            print('have not database')
            continue


        val_data = np.array(row)[:, 0]
        val_y = np.array(row)[:, 1]

        #room_name = np.array(row)[:, 2]
        date = np.array(row)[:, 3]
        val_labels = {row[i][0]: row[i][1] for i in range(len(row))}
        val_time = np.array(row)[:, 4]

        labels.update(val_labels)

        if len(val_data) < 1000:
            continue


        partition2 = {'train': training_data, 'val': val_data}

        validation_generator2 = DataGenerator(partition2['val'], labels, **val_params, path='')

        pred = model.predict_generator(generator=validation_generator2)
        #result = model.evaluate_generator(generator=validation_generator2)
        tmppred = pred[:]

        import scipy.signal

        pred = scipy.signal.savgol_filter(tmppred.flatten(),449,0)
        val_y = val_y.flatten()
        val_y_filtered = scipy.signal.savgol_filter(val_y,449,0)

        mae = sklearn.metrics.mean_absolute_error(val_y_filtered.astype('float32'),pred)
        mse = sklearn.metrics.mean_squared_error(val_y_filtered.astype('float32'), pred)

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


        #for j in
        array_bae_mean[output_cnt],array_bae_diff[output_cnt] = bland_altman_result(pred.flatten()[1::20], val_y_filtered.astype('float32')[1::20])
        output_cnt = output_cnt+1
        bland_altman_plot_20seg(pred.flatten(), val_y_filtered.astype('float32'))


        with open(savepath + room + "_" + val_file + 'SV' + '.csv', 'w',
                  newline='') as corr_csv:
            wr = csv.writer(corr_csv)
            wr.writerow(['room','file','pred_sv','compared_sv'])
            for j in range(len(pred)):
                wr.writerow([room,val_file,pred[j],val_y_filtered[j]])

        with open(savepath + 'all_data.csv', 'a',
                  newline='') as corr_csv:
            wr = csv.writer(corr_csv)
            #wr.writerow(['room','file','time','pred_svv','compared_svv'])
            for j in range(len(pred)):
                wr.writerow([room,val_file,pred[j],val_y_filtered[j]])


        with open(savepath + 'result.csv', 'a',
                  newline='') as corr_csv:
            wr = csv.writer(corr_csv)
            wr.writerow([room,val_file,str(mae)[:5],str(mse)[:5],str(corr)[:4]])

import itertools

diff = []
for i in range(len(array_bae_diff)):
    diff = diff + list(array_bae_diff[i])[:]


c_lst = [plt.cm.rainbow(a) for a in np.linspace(0.0, 1.0, len(array_bae_mean))]
md = np.mean(diff)  # Mean of the difference
sd = np.std(diff)  # Standard deviation of the difference
fig = plt.figure(figsize=(20, 10))
plt.axhline(md, color='gray', linestyle='--')
plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')
for i in range(len(c_lst)):
    if len(array_bae_diff[i]) == 1:
        continue
    plt.scatter(array_bae_mean[i],array_bae_diff[i],c=c_lst[i%30],s=3)


plt.title("Total_Bland_Altman plot  md : " +str(round(md,2)) +'  + : ' +str(round(md + 1.96 * sd,2)) + '  - : '+ str(round(md - 1.96 * sd,2)))
#plt.show()
plt.savefig(
    savepath +'ALL_BA' + '.png')
plt.close()


fig = plt.figure(figsize=(20, 10))

for i in range(1,len(array_bae_diff)):
    if len(array_bae_diff[i]) == 1:
        continue


    plt.subplot(5,7,i)
    plt.scatter(array_bae_mean[i], array_bae_diff[i], c=c_lst[i], s=3)
    plt.axhline( np.mean(array_bae_diff[i]), color='gray', linestyle='--')
    plt.axhline( np.mean(array_bae_diff[i]) + 1.96 *  np.std(array_bae_diff[i]), color='gray', linestyle='--')
    plt.axhline( np.mean(array_bae_diff[i]) - 1.96 *  np.std(array_bae_diff[i]), color='gray', linestyle='--')
    plt.savefig(
        savepath  + 'ALL_BA2' + '.png')


