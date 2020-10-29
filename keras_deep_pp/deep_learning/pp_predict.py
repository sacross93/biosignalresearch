from utils.my_classes import DataGenerator
from keras import models
from keras import layers
import numpy as np
from keras import optimizers

from keras.utils.training_utils import multi_gpu_model
def cnn2d_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(20000, 100,1)))
    model.add(layers.MaxPooling2D((5,2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((5, 1)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    #model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))

    return model
# Parameters
params = {'dim': (20000,100),
          'batch_size': 10,
          'n_classes': 1,
          'n_channels': 1,
          'shuffle': False}




def sort_file(files):
    result = []
    for i in range(len(files)):
        for file in files:
            if file.find('_'+str(i)+'.npy')>=0:
                result.append(file)
                #print(file)

    return result


import os

# Datasets

#filenames = ['180313','180322','180327','180605','180607','180608','180612','180614','180619','180621','180626','180626','180628','180629']
filenames = ['180322','180327','180607','180608','180612','180614','180619','180621','180626','180626','180628','180629']
#filenames = ['180628','180629']

for filename in filenames:
    filename = filename +'_2'

    ydata = os.listdir('/home/jmkim/ydata/')

    ID = os.listdir('/home/jmkim/data/')
    files = []
    for file in ID:
        if file.find(filename)>=0:
            files.append(file)

    x_data = sort_file(files)

    if len(x_data)<30:
        continue

    ypath = '/home/jmkim/ydata/'
    yfiles = os.listdir(ypath)
    y = []
    for file in yfiles:
        if file.find(filename)>=0:
            y_data = np.load(ypath+file)
            print(ypath+file)
            break

    print(len(x_data))
    print(len(y_data))
    labels = {x_data[i] : y_data[i] for i in range(len(y_data)) }




    partition = {'train' : x_data}
    #labels = {'180313_0_1' : 57.08444, '180313_0_2' :75.627174, '180313_0_0' : 73.48143 }
    #labels = {'180313_0_1' : 1, '180313_0_2' :0, '180313_0_0' : 1 }
    #[57.08444 , 75.627174, 73.48143]

    # Generators
    training_generator = DataGenerator(partition['train'], labels, **params)
    #validation_generator = DataGenerator(partition['validation'], labels, **params)

    # Design model
    model = cnn2d_model()
    [...] # Architecture
    opt = optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model = multi_gpu_model(model, gpus=4)
    #model.load_weights('/home/projects/pcg_transform/pcg_AI/deep/PP/saved_weight/keras_19_21_deep_PP.h5')
    model.compile(loss='mse', optimizer=opt, metrics=['mse'])
    # Train model on dataset
    model.fit_generator(generator=training_generator,epochs=1000)

    model.save_weights('/home/projects/pcg_transform/pcg_AI/deep/PP/saved_weight/' + filename + '_predict_PP.h5')

    """
    
    pred = model.predict_generator(generator=training_generator)
    result = model.evaluate_generator(generator=training_generator)


    corr = np.corrcoef(pred.flatten(), y_data)[0, 1]
    plt.title("_Mean_square_error :" + str(result[1])[:4]+ "  corr :" + str(corr)[:4])
    plt.plot(pred, 'b', label="Predicted")
    plt.plot(y_data, 'r', label='PP')
    plt.legend()
    plt.savefig('/home/projects/pcg_transform/pcg_AI/deep/PP/result/deep_to_PP_100_distinct/_21_29_to' + filename + '_predict_PP.png')
    plt.show()

   """

    """
    
    # Train model on dataset
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=False,
                        workers=1)
    
    """