import keras
from sklearn.metrics import mean_squared_error
import scipy.signal
import matplotlib.pyplot as plt
from SVV_projects.read_SVV import read_abp_svv_10sec, read_abp_sv_10sec
from tcn.tcn import compiled_tcn
import numpy as np

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "3";


save_path = '/home/projects/pcg_transform/pcg_AI/deep/SVV/result/self_graph/'

#for filedate in files:
files=[]
for i in range(31):
    tmp = 180302+i
    files.append(str(tmp))

for filedate in files:
    x_data, y_data = read_abp_svv_10sec([filedate])
    training_data =x_data[:len(x_data)*2//3]
    ty = y_data[:len(x_data)*2//3]

    #if len(training_data) == 0:
    #    continue

    # validation
    val_filedate = filedate

    validation_data = x_data[len(x_data)*2//3:]
    val_y = y_data[len(x_data)*2//3:]



    class PrintSomeValues(keras.callbacks.Callback):

        def on_epoch_begin(self, epoch, logs={}):
            print(f'x_test[0:1] = {validation_data[0:1]}.')
            print(f'y_test[0:1] = {val_y[0:1]}.')
            print(f'pred = {self.model.predict(validation_data[0:1])}.')



    model = compiled_tcn_v2(return_sequences=False,
                         num_feat=training_data.shape[2],
                         num_classes=0,
                         nb_filters=64,
                         kernel_size=3,
                         dilations=[2 ** i for i in range(9)],
                         nb_stacks=2,
                         max_len=training_data.shape[1],
                         activation='norm_relu',
                         use_skip_connections=True,
                         regression=True,
                         dropout_rate=0)

    print(f'x_train.shape = {training_data.shape}')
    print(f'y_train.shape = {validation_data.shape}')

    psv = PrintSomeValues()

    # Using sparse softmax.
    # http://chappers.github.io/web%20micro%20log/2017/01/26/quick-models-in-keras/

    model.fit(training_data, ty, validation_data=(validation_data, val_y), epochs=1,
               batch_size=128)





    pred = model.predict(validation_data)
    result = model.evaluate(validation_data, val_y)
    tmppred = pred[:]

    import scipy.signal

    pred = scipy.signal.savgol_filter(tmppred.flatten(), 37, 0)
    val_y = val_y.flatten()
    filtered_val_y = scipy.signal.savgol_filter(val_y.flatten(), 37, 0)
    corr = np.corrcoef(pred, filtered_val_y.astype('float32'))[0, 1]
    # corr = np.corrcoef(pred.flatten(), val_y.astype('float32'))[0, 1]
    plt.title( "  corr :" + str(corr)[:4])
    plt.plot(pred, 'b', label="Predicted")
    plt.plot(filtered_val_y, 'r', label='SVV')
    # plt.xticks([i * (len(pred) // 5 - 1) for i in range(6)])
    plt.legend()
    #plt.savefig(
    #    '/home/projects/pcg_transform/pcg_AI/deep/SVV/result/self_graph_resnet01/' + filedate + 'val_SVV_self_tt' + '.png')
    plt.show()

