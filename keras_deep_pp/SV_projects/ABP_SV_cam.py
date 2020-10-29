from utils.my_classes import DataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
#from utils.processing import *
from utils.models import *
import csv
import keras

from keras.utils.training_utils import multi_gpu_model

import sklearn.metrics
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D,Dense,LSTM,Dropout
import pymysql
import os
from tcn.tcn import  compiled_tcn
import scipy.signal


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
import datetime
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "3";



folder_name = 'model_sv_cam_test/'

savepath = '/home/projects/pcg_transform/pcg_AI/deep/SV/result/'+folder_name

if not os.path.isdir(savepath ):
    os.mkdir(savepath )


stdate = 180201
enddate = 180229

valstdate = 180301
valenddate = 180311

conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                       db='abp_svv_generator', charset='utf8')
curs = conn.cursor()

sql = """select file_name,VG_SV from abp_svv_generator.abp_sv_small where date >=%s and date <=%s order by date,file_name;"""
curs.execute(sql,(stdate,enddate))
row = curs.fetchall()

training_data = np.array(row)[:,0]
labels = {row[i][0]: row[i][1] for i in range(len(row))}



#sql = """select file_name,VG_SV from abp_svv_generator.abp_sv_small where date >=%s and date <=%s order by date,file_name;"""

curs.execute(sql,(valstdate,valenddate))
row = curs.fetchall()

val_data = np.array(row)[:,0]
val_y = np.array(row)[:,1]
val_labels = {row[i][0]: row[i][1] for i in range(len(row))}

conn.close()

labels.update(val_labels)


#
# files = ['180301','180303','180304','180305','180307']
# training_files =files[:3]
# validation_files=[files[3]]
# test_files=[files[4]]
#


""""""
filedate = str(stdate)

""""""


params = {'dim': (300,),
          'batch_size': 32,
          'n_classes': 1,
          'n_channels': 1,
          'shuffle': False}

val_params = {'dim': (300,),
              'batch_size': 32,
              'n_classes': 1,
              'n_channels': 1,
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



model = semi_vggnet1000_stride_1c()

#model = multi_gpu_model(model, gpus=2)
[...]  # Architecture
adam = optimizers.Adam(lr=0.000002, clipnorm=1.)
model.compile(adam, loss='mean_squared_error')

# Train model on dataset
#model.load_weights('/home/jmkim/PycharmProjects/keras_deep_pp/tmp.h5')


bestpath = savepath + filedate + '.h5'
#model.load_weights(bestpath)

#model. load_weights(bestpath)
epoch = 50

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

#model.save_weights('tmp_all_march_stride.h5')
#model.load_weights('tmp.h5')

model.load_weights(bestpath)
# model.load_weights(bestpath + filedate + 'best_resnet152_predict_SV_test.h5')
#model.save_weights('tmp_all_3month_vgglike_stride_ver2.h5')
#model.save_weights('tmp_all_3month_vgglike_stride2.h5')
#model.load_weights('tmp_all_3month_vgglike_stride2.h5')




inputdata = np.load(val_data[len(val_data) // 10])

#inputdata = np.stack([inputdata,np.load(val_data[len(val_data)*2 // 10])])

#try using vis
import vis

# from vis.visualization import visualize_cam
#
# visualize_cam(model,'conv_end',None,inputdata)


def grad_cam(model,inputdata):
    inputdata = np.reshape(inputdata, [1, 300, 1])

    #try using book
    model_output = model.output[:,0]
    last_conv_layer= model.get_layer('conv1d_1')
    grads = K.gradients(model_output,last_conv_layer.output)[0]
    pooled_grads = K.mean(grads,axis=(0,1))

    iterate = K.function([model.input],
                         [pooled_grads,last_conv_layer.output])

    pooled_grads_value,conv_layer_output_value = iterate([inputdata])



    for i in range(len(pooled_grads_value)):
        conv_layer_output_value[:,:,i] *= pooled_grads_value[i]

    grad_cam = np.average(conv_layer_output_value, 0)


    cam_data = []
    for i in range(len(grad_cam)):
        cam_data.append(np.average(grad_cam[i,:]))
    cam_data = np.reshape(cam_data,[1,len(cam_data)])

    from scipy.signal import savgol_filter
    #test_cam2 = savgol_filter(cam_data, 3,0)

    test_cam2 = np.resize(cam_data,[1,300])

    fig = plt.figure(figsize=(20, 10))
    ax0 = plt.subplot2grid((1, 1), (0, 0), colspan=1)
    plt.yticks(fontsize=15)
    ax0.plot(inputdata.flatten(),c='blue')
    ax0_2 = ax0.twinx()
    ax0_2.imshow(test_cam2,cmap='gist_heat',aspect='auto',alpha=0.4)

    plt.show()


    return test_cam2

def draw_10_graph(model,npydatapath,savename):


    fig = plt.figure(figsize=(20, 10))
    ax0 = plt.subplot2grid((2, 5), (0, 0), colspan=1)
    ax1 = plt.subplot2grid((2, 5), (0, 1), colspan=1)
    ax2 = plt.subplot2grid((2, 5), (0, 2), colspan=1)
    ax3 = plt.subplot2grid((2, 5), (0, 3), colspan=1)
    ax4 = plt.subplot2grid((2, 5), (0, 4), colspan=1)
    ax5 = plt.subplot2grid((2, 5), (1, 0), colspan=1)
    ax6 = plt.subplot2grid((2, 5), (1, 1), colspan=1)
    ax7 = plt.subplot2grid((2, 5), (1, 2), colspan=1)
    ax8 = plt.subplot2grid((2, 5), (1, 3), colspan=1)
    ax9 = plt.subplot2grid((2, 5), (1, 4), colspan=1)


    output_data1 = np.load(npydatapath[len(npydatapath) // 10])
    ax0.plot(output_data1,c='blue')
    ax0_2 = ax0.twinx()
    ax0_2.imshow(grad_cam(model,output_data1),cmap='gist_heat',aspect='auto',alpha=0.4)
    ax0.set(xlabel = len(npydatapath) // 10)

    output_data1 = np.load(npydatapath[len(npydatapath)* 2 // 10])
    ax1.plot(output_data1,c='blue')
    ax1_2 = ax1.twinx()
    ax1_2.imshow(grad_cam(model,output_data1),cmap='gist_heat',aspect='auto',alpha=0.4)
    ax1.set(xlabel = len(npydatapath)*2 // 10)

    output_data1 = np.load(npydatapath[len(npydatapath)* 3 // 10])
    ax2.plot(output_data1,c='blue')
    ax2_2 = ax2.twinx()
    ax2_2.imshow(grad_cam(model,output_data1),cmap='gist_heat',aspect='auto',alpha=0.4)
    ax2.set(xlabel = len(npydatapath)*3 // 10)

    output_data1 = np.load(npydatapath[len(npydatapath)* 4 // 10])
    ax3.plot(output_data1,c='blue')
    ax3_2 = ax3.twinx()
    ax3_2.imshow(grad_cam(model,output_data1),cmap='gist_heat',aspect='auto',alpha=0.4)
    ax3.set(xlabel = len(npydatapath)*4 // 10)

    output_data1 = np.load(npydatapath[len(npydatapath)* 5 // 10])
    ax4.plot(output_data1,c='blue')
    ax4_2 = ax4.twinx()
    ax4_2.imshow(grad_cam(model,output_data1),cmap='gist_heat',aspect='auto',alpha=0.4)
    ax4.set(xlabel = len(npydatapath)*5 // 10)

    output_data1 = np.load(npydatapath[len(npydatapath)* 6 // 10])
    ax5.plot(output_data1,c='blue')
    ax5_2 = ax5.twinx()
    ax5_2.imshow(grad_cam(model,output_data1),cmap='gist_heat',aspect='auto',alpha=0.4)
    ax5.set(xlabel = len(npydatapath)*6 // 10)

    output_data1 = np.load(npydatapath[len(npydatapath)* 7 // 10])
    ax6.plot(output_data1,c='blue')
    ax6_2 = ax6.twinx()
    ax6_2.imshow(grad_cam(model,output_data1),cmap='gist_heat',aspect='auto',alpha=0.4)
    ax6.set(xlabel = len(npydatapath)*7 // 10)

    output_data1 = np.load(npydatapath[len(npydatapath)* 8 // 10])
    ax7.plot(output_data1,c='blue')
    ax7_2 = ax7.twinx()
    ax7_2.imshow(grad_cam(model,output_data1),cmap='gist_heat',aspect='auto',alpha=0.4)
    ax7.set(xlabel = len(npydatapath)*8 // 10)

    output_data1 = np.load(npydatapath[len(npydatapath)* 9 // 10])
    ax8.plot(output_data1,c='blue')
    ax8_2 = ax8.twinx()
    ax8_2.imshow(grad_cam(model,output_data1),cmap='gist_heat',aspect='auto',alpha=0.4)
    ax8.set(xlabel = len(npydatapath) *9 // 10)

    output_data1 = np.load(npydatapath[len(npydatapath) -1])
    ax9.plot(output_data1,c='blue')
    ax9_2 = ax9.twinx()
    ax9_2.imshow(grad_cam(model,output_data1),cmap='gist_heat',aspect='auto',alpha=0.4)
    ax9.set(xlabel = len(npydatapath) -1)
    #plt.show()


    plt.savefig(
        savename + '.png')
    # plt.show()
    plt.close()


plt.plot(val_mse, 'b', label='Validation mean_square_error')
plt.title(filedate + 'vgglike Training and validation mean_square_error')
plt.legend()
# plt.show()
#plt.savefig(
#    '/home/projects/pcg_transform/pcg_AI/deep/SVV/result/self_graph_tcn01/' + filedate + 'loss_val_SVV_self_128_8_t2' + '.png')
plt.show()
plt.close()


val_params = {'dim': (300,),
              'batch_size': 1,
              'n_classes': 1,
              'n_channels': 1,
              'shuffle': False}


validation_files = range(180301, 180332)

for i in range(len(validation_files)):

    val_file = validation_files[i]
    print('start : ', val_file)

    conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                           db='abp_svv_generator', charset='utf8')
    curs = conn.cursor()


    sql = """select distinct room_name from abp_svv_generator.abp_sv_small2 where date =%s;"""
    curs.execute(sql, (val_file))
    rooms = curs.fetchall()

    conn.close()
    print(rooms)



    if len(rooms)<1:
        continue

    for room in np.array(rooms):
        room = list(room)[0]

        conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                               db='abp_svv_generator', charset='utf8')
        curs = conn.cursor()

        sql = """select file_name,VG_SV,room_name,date,time from abp_svv_generator.abp_sv_small2 where date =%s and room_name = %s order by date,file_name;"""
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
        timestamp = np.array(row)[:, -1]
        val_labels = {row[i][0]: row[i][1] for i in range(len(row))}

        labels.update(val_labels)

        if len(val_data) < 1000:
            continue

        timedate = []
        for time in timestamp:
            timedate.append((datetime.datetime.min +time).time())

        partition2 = {'train': training_data, 'val': val_data}


        validation_generator2 = DataGenerator(partition2['val'], labels, **val_params, path='')

        pred = model.predict_generator(generator=validation_generator2)
        #result = model.evaluate_generator(generator=validation_generator2)
        tmppred = pred[:]



        pred = scipy.signal.savgol_filter(tmppred.flatten(),899,0)
        val_y = val_y.flatten()
        val_y_filtered = scipy.signal.savgol_filter(val_y,899,0)

        mae = sklearn.metrics.mean_absolute_error(val_y_filtered.astype('float32'),pred)

        corr = np.corrcoef(pred.flatten(), val_y_filtered.astype('float32'))[0, 1]


        fig = plt.figure(figsize=(20, 10))
        plt.title(room + "_" +str(val_file) + "_Mean_absolued_error :" + str(mae)[:5] + "  corr :" + str(corr)[:4])
        plt.plot(pred.flatten(), 'b', label="Predicted")
        plt.plot(val_y_filtered, 'r', label='SV')
        #plt.xticks([i * (len(pred) // 5 - 1) for i in range(6)])
        plt.legend(fontsize='xx-large')


        savename = savepath + room + "_" + str(val_file) + 'SV_test300_test'

        plt.savefig(
           savename + '.png')
        plt.close()


        #draw_10_graph(model,val_data,savename + '_CAM')




        """
        with open(savepath + 'result.csv', 'a',
                  newline='') as corr_csv:
            wr = csv.writer(corr_csv)
            wr.writerow([room,val_file,str(mae)[:5],str(corr)[:4]])
        """




"""
#trying using other
def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

class_weights = model.layers[-1].get_weights()[0]
final_conv_layer = get_output_layer(model, "conv1d_3")
get_output = K.function([model.layers[0].input], \
                        [final_conv_layer.output,
                         model.layers[-1].output])



[conv_outputs, predictions] = get_output([inputdata])
conv_outputs = conv_outputs[0,:,:]

cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[1:3])
target_class = 0
for i, w in enumerate(class_weights[:, target_class]):
    cam += w * conv_outputs[i, :]
"""
