from utils.my_classes import DataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from utils.processing import get_ml_pp_data

from keras.utils.training_utils import multi_gpu_model

#model = resnet152_model()
#model.summary()



filedate = '180628'
#filedate = '180628'

x_data,y_data = get_ml_pp_data(filedate)

training_data = x_data[:len(x_data)*2//3]
validation_data = x_data[len(x_data)*2//3:len(x_data)*5//6]
test_data = x_data[len(x_data)*5//6:]


ty = y_data[:len(y_data)*2//3]
val_y = y_data[len(y_data)*2//3:len(y_data)*5//6]
test_y = y_data[len(y_data)*5//6:]
"""
val_y_time = y_time[len(data)*2//3:len(data)*5//6]
test_y_time = y_time[len(data)*5//6:]
"""

from keras.models import Sequential
from keras.layers import Dense
import keras

# Design model
model = Sequential()
model.add(keras.layers.Dense(64, activation='relu',
                       input_shape= (x_data.shape[1],)))
model.add(keras.layers.Dense(64,activation='relu'))
model.add(keras.layers.Dense(1))




[...] # Architecture
opt = optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model = multi_gpu_model(model, gpus=4)

#model.load_weights('/home/projects/pcg_transform/pcg_AI/deep/PP/save_weight/resnet152_predict_SV.h5')

#model.load_weights('/home/projects/pcg_transform/pcg_AI/deep/PP/save_weight/vgg16_best_predict_PP.h5')
model.compile(loss='mse', optimizer=opt, metrics=['mse'])
# Train model on dataset



bestpath = '/home/projects/pcg_transform/pcg_AI/ml/PP/result/saved_weight/'+filedate+'best_ml_predict_PP.h5'
#checkpoint = keras.callbacks.ModelCheckpoint(bestpath, monitor='val_mse',verbose=1,save_best_only=True,mode='min')
"""
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
"""

#checkpoint = keras.callbacks.ModelCheckpoint(filepath=bestpath, verbose=1,save_weights_only=True, save_best_only=True)

#callbacks_list = [checkpoint]





epoch = 10

best = 99999
comple_flag = 0
save_flag=0
val_mse = []



for j in range(100):
    if comple_flag ==1:
        break
    if j >1:
        save_flag=1
    for i in range(epoch):
        print('epoch : '+str((i+1)+(j)*10))
        history = model.fit(training_data,ty,validation_data=(validation_data,val_y))
        mse = history.history['val_mean_squared_error'][0]
        val_mse.append(mse)
        print(mse)
        if mse <best and save_flag==1:
            print('saved_best_data')
            best = mse
            model.save_weights(
                bestpath)



        if mse >best*3:
            print('Training_complete')
            comple_flag =1
            break




model.load_weights(bestpath)
#model.load_weights(bestpath + filedate + 'best_resnet152_predict_SV_test.h5')

#model.save_weights('/home/projects/pcg_transform/pcg_AI/deep/SV/save_weight/'+filedate+'resnet152_predict_SV_test.h5')




mse = history.history['mean_squared_error']
val_mse = history.history['val_mean_squared_error']

epochs = range(len(mse))

plt.close(1)  # To clear previous figure
plt.close(2)

#plt.ylim(0,300)
#plt.plot( 'bo', label='Training mean_square_error')
plt.plot(val_mse, 'b', label='Validation mean_square_error')
plt.title(filedate+'Training and validation mean_square_error')
plt.legend()
plt.show()
#plt.savefig('/home/projects/pcg_transform/pcg_AI/deep/PP/saved_weight/keras_cnn_test_190109_deep_PP_mse.png')
#plt.figure()



pred = model.predict(validation_data)
result = model.evaluate(validation_data,val_y)
tmppred = pred[:]

corr = np.corrcoef(pred.flatten(),val_y.astype('float32'))[0, 1]
plt.title(str('val')+filedate+"_Mean_square_error :" + str(result[1])[:4]+ "  corr :" + str(corr)[:4])
plt.plot(pred, 'b', label="Predicted")
plt.plot( val_y, 'r', label='PP')
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

pred = model.predict(test_data)
result = model.evaluate(test_data,test_y)
tmppred = pred[:]

corr = np.corrcoef(pred.flatten(),test_y.astype('float32'))[0, 1]
plt.title(filedate+"_Mean_square_error :" + str(result[1])[:4]+ "  corr :" + str(corr)[:4])
plt.plot(pred, 'b', label="Predicted")
plt.plot( test_y, 'r', label='PP')
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