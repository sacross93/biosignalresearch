import keras
from utils.my_classes import DataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from processing import cnn2d_model, get_data, get_test_data

from keras.utils.training_utils import multi_gpu_model
from resnet152 import resnet152_model
model = resnet152_model()
model.summary()

params = {'dim': (20000,100),
          'batch_size': 10,
          'n_classes': 1,
          'n_channels': 1,
          'shuffle': False}

val_params = {'dim': (20000,100),
          'batch_size': 1,
          'n_classes': 1,
          'n_channels': 1,
          'shuffle': False}



# Datasets

#filenames = ['180313','180322','180327','180605','180607','180608','180612','180614','180619','180621','180626','180626','180628','180629']
filenames = ['180322','180327','180607','180608','180612','180614','180619','180621','180626','180628','180629']

validation_files = []
training_files = []
test_files = []
for i in range(len(filenames)):
    if i == 3 or i == 6:
        validation_files.append(filenames[i])
        continue
    if i == 9 or i == 12:
        test_files.append(filenames[i])
        continue

    training_files.append(filenames[i])




training_data, labels = get_data(training_files)
validation_data, val_labels = get_data(validation_files)
test_data, test_labels = get_data(test_files)



labels.update(val_labels)
labels.update(test_labels)




partition = {'train' : training_data, 'validation' : validation_data, 'test' : test_data}
#labels = {'180313_0_1' : 57.08444, '180313_0_2' :75.627174, '180313_0_0' : 73.48143 }
#labels = {'180313_0_1' : 1, '180313_0_2' :0, '180313_0_0' : 1 }
#[57.08444 , 75.627174, 73.48143]

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **val_params)
test_generator = DataGenerator(partition['test'], labels, **val_params)


# Design model
model = cnn2d_model()
[...] # Architecture
opt = optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model = multi_gpu_model(model, gpus=4)

#model.load_weights('/home/projects/pcg_transform/pcg_AI/deep/PP/save_weight/vgg16_best_predict_PP.h5')
model.compile(loss='mse', optimizer=opt, metrics=['mse'])
# Train model on dataset



bestpath = '/home/projects/pcg_transform/pcg_AI/deep/PP/save_weight/vgg16_best_predict_PP.h5'
checkpoint = keras.callbacks.ModelCheckpoint(bestpath, monitor='val_mse',verbose=1,save_best_only=True,mode='min')
callbacks_list = [checkpoint]


history = model.fit_generator(generator=training_generator,epochs=1000,validation_data=validation_generator
                    ,validation_steps=20,callbacks=[checkpoint])

model.save_weights('/home/projects/pcg_transform/pcg_AI/deep/PP/save_weight/vgg16_predict_PP2.h5')




mse = history.history['mean_squared_error']
val_mse = history.history['val_mean_squared_error']

epochs = range(len(mse))

plt.close(1)  # To clear previous figure
plt.close(2)

#plt.ylim(0,300)
plt.plot(epochs, mse, 'bo', label='Training mean_square_error')
plt.plot(epochs, val_mse, 'b', label='Validation mean_square_error')
plt.title('Training and validation mean_square_error')
plt.legend()
plt.show()
#plt.savefig('/home/projects/pcg_transform/pcg_AI/deep/PP/saved_weight/keras_cnn_test_190109_deep_PP_mse.png')
#plt.figure()


test01 = [validation_files[1]]
test01 = [test_files[0]]
#test01 = [training_files[0]]
test_data, test_labels,y_data = get_test_data(test01)
partition = {'test01': test_data}
test_generator = DataGenerator(partition['test01'], labels, **val_params)



pred = model.predict_generator(generator=test_generator)
result = model.evaluate_generator(generator=test_generator)
tmppred = pred[:]

corr = np.corrcoef(pred.flatten(), y_data)[0, 1]
plt.title(str(test01[0])+"_Mean_square_error :" + str(result[1])[:4]+ "  corr :" + str(corr)[:4])
plt.plot(pred, 'b', label="Predicted")
plt.plot(y_data, 'r', label='PP')
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