from utils.my_classes import DataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
#from processing import get_sv_data
from PCG_SVV.read_data import *
from PCG_SVV.pcg_models import *
from keras.utils.training_utils import multi_gpu_model
from tcn.tcn import *
#from resnet152 import resnet152_model

# model = resnet152_model()
# model.summary()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3";


params = {'dim': (20000,),
          'batch_size': 16,
          'n_classes': 1,
          'n_channels': 5,
          'shuffle': False}

val_params = {'dim': (20000,),
              'batch_size': 1,
              'n_classes': 1,
              'n_channels': 5,
              'shuffle': False}

# filedate = '180607'
filedate = '190215'
datapath = '/home/projects/pcg_transform/pcg_AI/deep/Data/'

data, labels, y_time, y_data = get_pcg_sv_data([filedate], x_path=datapath,y='SV')

training_data = data

ty = np.array(y_data)

partition = {'train': training_data, 'validation': training_data}
# labels = {'180313_0_1' : 57.08444, '180313_0_2' :75.627174, '180313_0_0' : 73.48143 }
# labels = {'180313_0_1' : 1, '180313_0_2' :0, '180313_0_0' : 1 }
# [57.08444 , 75.627174, 73.48143]


# Generators
training_generator = DataGenerator(partition['train'], labels, **params, path=datapath)
validation_generator = DataGenerator(partition['validation'], labels, **val_params, path=datapath)
#test_generator = DataGenerator(partition['test'], labels, **val_params, path=datapath)

# Design model
model = compiled_tcn(return_sequences=False,
                        num_feat=5,
                        num_classes=0,
                        nb_filters=128,
                        kernel_size=8,
                        dilations=[2 ** i for i in range(9)],
                        nb_stacks=2,
                        max_len=20000,
                        activation='norm_relu',
                        use_skip_connections=True,
                        regression=True,
                        dropout_rate=0)

[...]  # Architecture
opt = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model = multi_gpu_model(model, gpus=2)

# model.load_weights('/home/projects/pcg_transform/pcg_AI/deep/PP/save_weight/resnet152_predict_SV.h5')

# model.load_weights('/home/projects/pcg_transform/pcg_AI/deep/PP/save_weight/vgg16_best_predict_PP.h5')
model.compile(loss='mse', optimizer=opt, metrics=['mse'])
# Train model on dataset


bestpath = '/home/projects/pcg_transform/pcg_AI/deep/PP/save_weight/best_resnet152_predict_SV.h5'
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

for j in range(3):
    if comple_flag == 1:
        break

    for i in range(epoch):
        if i > 1:
            save_flag = 1
        print('epoch : ' + str((i + 1) + (j) * 10))
        history = model.fit_generator(generator=training_generator, epochs=1, validation_data=validation_generator)
        mse = history.history['val_loss'][0]
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
# model.load_weights('tmp.h5')

model.load_weights(bestpath)



epoch = 5

best = 99999
comple_flag = 0
save_flag = 0
for j in range(3):
    if j > 0:
        save_flag = 1
    for i in range(epoch):
        print((i + 1) * (j + 1))
        history = model.fit_generator(generator=training_generator, epochs=1 )
        """
        mse = history.history['val_mean_squared_error']
        print(mse[0])
        if mse[0] < best and save_flag == 1:
            print('saved_best_data')
            best = mse[0]
            model.save_weights(
                bestpath + filedate + 'best_resnet152_predict_SV_test.h5')

        if mse[0] > best * 3:
            print('Training_complete')
            break
        """
model.save_weights('tmp.h5')

#model.load_weights(bestpath + '180607' + 'best_resnet152_predict_SV_test.h5')
# model.load_weights(bestpath + filedate + 'best_resnet152_predict_SV_test.h5')

# model.save_weights('/home/projects/pcg_transform/pcg_AI/deep/SV/save_weight/'+filedate+'resnet152_predict_SV_test.h5')

"""
mse = history.history['mean_squared_error']
val_mse = history.history['val_mean_squared_error']

epochs = range(len(mse))

plt.close(1)  # To clear previous figure
plt.close(2)

# plt.ylim(0,300)
plt.plot(mse, 'bo', label='Training mean_square_error')
plt.plot(val_mse, 'b', label='Validation mean_square_error')
plt.title(filedate + 'Training and validation mean_square_error')
plt.legend()
plt.show()
# plt.savefig('/home/projects/pcg_transform/pcg_AI/deep/PP/saved_weight/keras_cnn_test_190109_deep_PP_mse.png')
# plt.figure()
"""
import scipy.signal

pred = model.predict_generator(generator=validation_generator)
result = model.evaluate_generator(generator=validation_generator)
tmppred = pred[:]
val_y = ty.flatten()

pred = scipy.signal.savgol_filter(tmppred.flatten(), 37, 0)
#val_y = val_y.flatten()
val_y_filtered = scipy.signal.savgol_filter(val_y, 37, 0)



corr = np.corrcoef(pred.flatten(), np.float32(val_y))[0, 1]
plt.title(str('val') + filedate + "_Mean_square_error :" + str(result[1])[:4] + "  corr :" + str(corr)[:4])
plt.plot(pred, 'b', label="Predicted")
plt.plot(val_y, 'r', label='SV')
plt.xticks([i * (len(pred) // 5 - 1) for i in range(6)])
plt.legend()
# plt.savefig('/home/projects/pcg_transform/pcg_AI/deep/PP/result/vgg16_pp/vgg16_predict_PP_val_'+str(test01[0])+'.png')
plt.show()
