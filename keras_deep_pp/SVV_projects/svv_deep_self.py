from utils.my_classes import DataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from utils.processing import get_ml_pp_data
import csv
from utils.models import semi_vggnet
from sklearn.metrics import mean_squared_error
import scipy.signal

from SVV_projects.read_SVV import read_abp_svv, read_abp_sv_10sec
from keras.utils.training_utils import multi_gpu_model
from utils.resnet1D import resnet1D_model

import keras

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";


# model = resnet152_model()
# model.summary()

#files = [ '180206', '180207', '180208', '180209',  '180213', '180214']
files=[]
for i in range(31):
    tmp = 180302+i
    files.append(str(tmp))

save_path = '/home/projects/pcg_transform/pcg_AI/deep/SVV/result/self_graph/'

for filedate in files:
#filedate = '180303'

    x_data, y_data = read_abp_svv([filedate])
    training_data =x_data[:len(x_data)*2//3]
    ty = y_data[:len(x_data)*2//3]

    #if len(training_data) == 0:
    #    continue

    # validation
    val_filedate = filedate

    validation_data = x_data[len(x_data)*2//3:]
    val_y = y_data[len(x_data)*2//3:]

    """
    val_y_time = y_time[len(data)*2//3:len(data)*5//6]
    test_y_time = y_time[len(data)*5//6:]
    """





    # Design model
    model = semi_vggnet()
    model.summary()

    #model.add(LSTM(128,activation='relu',return_sequences=True,dropout=0.5))


    [...]  # Architecture
    opt = optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    #model = multi_gpu_model(model, gpus=4)

    # model.load_weights('/home/projects/pcg_transform/pcg_AI/deep/PP/save_weight/resnet152_predict_SV.h5')

    # model.load_weights('/home/projects/pcg_transform/pcg_AI/deep/PP/save_weight/vgg16_best_predict_PP.h5')
    model.compile(loss='mse', optimizer=opt, metrics=['mse'])
    # Train model on dataset

    bestpath = '/home/projects/pcg_transform/pcg_AI/deep/SVV/saved_weight/' + filedate + 'vgglike_predict_SVV.h5'
    bestcorrpath = '/home/projects/pcg_transform/pcg_AI/deep/SVV/saved_weight/' + filedate + 'vgglike_predict_SVV.h5'
    # checkpoint = keras.callbacks.ModelCheckpoint(bestpath, monitor='val_mse',verbose=1,save_best_only=True,mode='min')
    """
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    """

    # checkpoint = keras.callbacks.ModelCheckpoint(filepath=bestpath, verbose=1,save_weights_only=True, save_best_only=True)

    # callbacks_list = [checkpoint]



    epoch = 10

    best = 99999
    comple_flag = 0
    save_flag = 0
    val_mse = []



    for j in range(5):
        if comple_flag == 1:
            break

        for i in range(epoch):
            if i > 1:
                save_flag = 1
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

            if mse > best * 10:
                print('Training_complete')
                comple_flag = 1
                break

    model.save_weights('tmp.h5')
    #model.load_weights('tmp.h5')

    model.load_weights(bestpath)


    plt.close(1)  # To clear previous figure
    plt.close(2)
    import scipy.signal
    #result_val_mse = scipy.signal.savgol_filter(val_mse,9,0)
    # plt.ylim(0,300)
    # plt.plot( 'bo', label='Training mean_square_error')
    plt.plot(val_mse, 'b', label='Validation mean_square_error')
    plt.title(filedate + 'Training and validation mean_square_error')
    plt.legend()
    #plt.show()
    plt.savefig(
        '/home/projects/pcg_transform/pcg_AI/deep/SVV/result/self_graph_vgglike/' + filedate +'loss_val_SVV_self' + '.png')
    plt.show()
    plt.close()
    # plt.figure()

    pred = model.predict(validation_data)
    result = model.evaluate(validation_data, val_y)
    tmppred = pred[:]




    import scipy.signal

    pred = scipy.signal.savgol_filter(tmppred.flatten(),37,0)
    val_y = val_y.flatten()
    filtered_val_y= scipy.signal.savgol_filter(val_y.flatten(),37,0)
    corr = np.corrcoef(pred, filtered_val_y.astype('float32'))[0, 1]
    #corr = np.corrcoef(pred.flatten(), val_y.astype('float32'))[0, 1]
    plt.title(str('val') + filedate + "_Mean_square_error :" + str(result[1])[:4] + "  corr :" + str(corr)[:4])
    plt.plot(pred, 'b', label="Predicted")
    plt.plot(filtered_val_y, 'r', label='SVV')
    #plt.xticks([i * (len(pred) // 5 - 1) for i in range(6)])
    plt.legend()
    plt.savefig(
        '/home/projects/pcg_transform/pcg_AI/deep/SVV/result/self_graph_vgglike/' + filedate + 'val_SVV_self_tt' + '.png')
    plt.show()



