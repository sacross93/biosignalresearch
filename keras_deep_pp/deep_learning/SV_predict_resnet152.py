from utils.my_classes import DataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from processing import get_sv_data

from keras.utils.training_utils import multi_gpu_model
from resnet152 import resnet152_model

#model = resnet152_model()
#model.summary()


params = {'dim': (5000,100),
          'batch_size': 8,
          'n_classes': 1,
          'n_channels': 1,
          'shuffle': False}

val_params = {'dim': (5000,100),
          'batch_size': 1,
          'n_classes': 1,
          'n_channels': 1,
          'shuffle': False}


#filedate = '180607'
filedate = '180628'
datapath = '/home/jmkim/data_5sec/'

data, labels, y_time,y_data = get_sv_data([filedate],x_path=datapath)

training_data = data[:len(data)*2//3]
validation_data = data[len(data)*2//3:len(data)*5//6]
test_data = data[len(data)*5//6:]

ty = y_data[:len(data)*2//3]
val_y = y_data[len(data)*2//3:len(data)*5//6]
test_y = y_data[len(data)*5//6:]
val_y_time = y_time[len(data)*2//3:len(data)*5//6]
test_y_time = y_time[len(data)*5//6:]



partition = {'train' : training_data, 'validation' : validation_data , 'test' : test_data }
#labels = {'180313_0_1' : 57.08444, '180313_0_2' :75.627174, '180313_0_0' : 73.48143 }
#labels = {'180313_0_1' : 1, '180313_0_2' :0, '180313_0_0' : 1 }
#[57.08444 , 75.627174, 73.48143]



# Generators
training_generator = DataGenerator(partition['train'], labels, **params,path = datapath )
validation_generator = DataGenerator(partition['validation'], labels, **val_params,path = datapath )
test_generator = DataGenerator(partition['test'], labels, **val_params,path = datapath )


# Design model
model = resnet152_model()

[...] # Architecture
opt = optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model = multi_gpu_model(model, gpus=4)

#model.load_weights('/home/projects/pcg_transform/pcg_AI/deep/PP/save_weight/resnet152_predict_SV.h5')

#model.load_weights('/home/projects/pcg_transform/pcg_AI/deep/PP/save_weight/vgg16_best_predict_PP.h5')
model.compile(loss='mse', optimizer=opt, metrics=['mse'])
# Train model on dataset



bestpath = '/home/projects/pcg_transform/pcg_AI/deep/PP/save_weight/best_resnet152_predict_SV.h5'
#checkpoint = keras.callbacks.ModelCheckpoint(bestpath, monitor='val_mse',verbose=1,save_best_only=True,mode='min')
"""
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
"""

#checkpoint = keras.callbacks.ModelCheckpoint(filepath=bestpath, verbose=1,save_weights_only=True, save_best_only=True)

#callbacks_list = [checkpoint]

from custom_callbacks import CustomHistory
custom_hist = CustomHistory()
custom_hist.init()


epoch = 10

best = 99999
comple_flag = 0
save_flag=0
for j in range(10):
    if j >1:
        save_flag=1
    for i in range(epoch):
        print((i+1)*(j+1))
        history = model.fit_generator(generator=training_generator,epochs=1,validation_data=validation_generator, validation_steps = 5
                                      , callbacks=[custom_hist])
        mse = history.history['val_mean_squared_error']
        print(mse[0])
        if mse[0] <best and save_flag==1:
            print('saved_best_data')
            best = mse[0]
            model.save_weights(
                bestpath + filedate + 'best_resnet152_predict_SV_test.h5')

        if mse[0] >best*3:
            print('Training_complete')
            break



#model.load_weights(bestpath + '180607' + 'best_resnet152_predict_SV_test.h5')
#model.load_weights(bestpath + filedate + 'best_resnet152_predict_SV_test.h5')

#model.save_weights('/home/projects/pcg_transform/pcg_AI/deep/SV/save_weight/'+filedate+'resnet152_predict_SV_test.h5')




mse = history.history['mean_squared_error']
val_mse = history.history['val_mean_squared_error']

epochs = range(len(mse))

plt.close(1)  # To clear previous figure
plt.close(2)

#plt.ylim(0,300)
plt.plot(custom_hist.train_mse, 'bo', label='Training mean_square_error')
plt.plot(custom_hist.val_mse, 'b', label='Validation mean_square_error')
plt.title(filedate+'Training and validation mean_square_error')
plt.legend()
plt.show()
#plt.savefig('/home/projects/pcg_transform/pcg_AI/deep/PP/saved_weight/keras_cnn_test_190109_deep_PP_mse.png')
#plt.figure()

np

pred = model.predict_generator(generator=training_generator)
result = model.evaluate_generator(generator=training_generator)
tmppred = pred[:]

corr = np.corrcoef(pred.flatten(),ty)[0, 1]
plt.title(str('val')+filedate+"_Mean_square_error :" + str(result[1])[:4]+ "  corr :" + str(corr)[:4])
plt.plot(pred, 'b', label="Predicted")
plt.plot( ty, 'r', label='SV')
plt.xticks([i*(len(pred)//5-1) for i in range(6)])
plt.legend()
#plt.savefig('/home/projects/pcg_transform/pcg_AI/deep/PP/result/vgg16_pp/vgg16_predict_PP_val_'+str(test01[0])+'.png')
plt.show()

"""
j = 0
while(len(pred)>j):
    if pred[j]>100:
        
        pred = np.delete(pred,j)
        val_y = np.delete(val_y,j)
        val_y_time = np.delete(val_y_time,j)
    j = j+1

"""

pred = model.predict_generator(generator=test_generator)
result = model.evaluate_generator(generator=test_generator)
tmppred = pred[:]

corr = np.corrcoef(pred.flatten(), test_y)[0, 1]
plt.title(str('test')+"_Mean_square_error :" + str(result[1])[:4]+ "  corr :" + str(corr)[:4])
plt.plot(test_y_time,pred, 'b', label="Predicted")
plt.plot(test_y_time,test_y, 'r', label='SV')
plt.xticks([i*(len(pred)//5-1) for i in range(6)])
plt.legend()
#plt.savefig('/home/projects/pcg_transform/pcg_AI/deep/PP/result/vgg16_pp/vgg16_predict_PP_val_'+str(test01[0])+'.png')
plt.show()
"""




# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    workers=1)

"""