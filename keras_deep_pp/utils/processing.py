from keras import models
from keras import layers
import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt

#기본 cnn 모델(5초)
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

#20초 cnn 망한 모
def vgg_cnn2d_model():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu',
                            input_shape=(20000, 100,1)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((5,2),strides=(2,2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((5, 1),strides=(2,2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 2),strides=(2,2)))


    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3),strides=(2,2)))

    #model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    #model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    #model.add(layers.MaxPooling2D((3, 1),strides=(2,2)))

    model.add(layers.Flatten())


    #model.add(layers.Dense(1273856, activation='relu'))
    model.add(layers.Dropout(0.5))

    #model.add(layers.Dense(1273856, activation='relu'))
    #model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='linear'))

    return model
# Parameters

#이름 이상할때 매칭해줬던거(이제 안씀)
def sort_file(files):
    result = []
    for i in range(len(files)):
        for file in files:
            if file.find('_'+str(i)+'.npy')>=0:
                result.append(file)
                #print(file)

    return result

#20초 data폴더에서 가져와서 pp와 매
def get_data(filenames):

    training_data = []
    labels = {}
    for filename in filenames:

        for z in range(3):
            filename = filename +'_'+str(z)

            import os
            ID = os.listdir('/home/jmkim/data/')
            files = []
            for file in ID:
                if file.find(filename)>=0:
                    files.append(file)

            x_data = sort_file(files)



            ypath = '/home/jmkim/ydata/'
            yfiles = os.listdir(ypath)
            y = []
            for file in yfiles:
                if file.find(filename)>=0:
                    y_data = np.load(ypath+file)
                    print(ypath+file)

                    break

            if len(x_data) != len(y_data):
                print(file)
                print('dismatching file')
                continue

            print(len(x_data))

            # x_data + x_data(connect)
            print(len(y_data))

            #tmp_label ={x_data[i] : y_data[i] for i in range(len(y_data)) }

            tmp_labels = {x_data[i] : y_data[i] for i in range(len(y_data)) }

            labels.update(tmp_labels)
            training_data = training_data + x_data
            len(training_data)
            len(labels)



    return training_data, labels


def get_test_data(filenames):

    result_y_data = np.array([])
    training_data = []
    labels = {}
    for filename in filenames:

        for z in range(3):
            filename = filename +'_'+str(z)

            import os
            ID = os.listdir('/home/jmkim/data/')
            files = []
            for file in ID:
                if file.find(filename)>=0:
                    files.append(file)

            x_data = sort_file(files)



            ypath = '/home/jmkim/ydata/'
            yfiles = os.listdir(ypath)
            y = []
            for file in yfiles:
                if file.find(filename)>=0:
                    y_data = np.load(ypath+file)
                    print(ypath+file)


                    break

            if len(x_data) != len(y_data):
                print(file)
                print('dismatching file')
                continue

            print(len(x_data))

            # x_data + x_data(connect)
            print(len(y_data))

            #tmp_label ={x_data[i] : y_data[i] for i in range(len(y_data)) }
            result_y_data = np.concatenate([result_y_data,y_data])
            tmp_labels = {x_data[i] : y_data[i] for i in range(len(y_data)) }

            labels.update(tmp_labels)
            training_data = training_data + x_data
            len(training_data)
            len(labels)



    return training_data, labels, result_y_data

#sv와 데이터를 매칭, 경로지정 등 기능 추가.
def get_sv_data(filenames,x_path='/home/jmkim/data2',delay=10):




    y_data = []
    result_y_data = np.array([])
    training_data = []
    labels = {}
    for filename in filenames:
        filename = filename

        import os
        ID = os.listdir(x_path)
        files = []
        for file in ID:
            if file.find(filename)>=0:
                files.append(file)

        #x_data = sort_file(files)



        ypath = '/home/projects/pcg_transform/pcg_AI/deep/SV/SV_data/D-06/'+filename+'/'+'feature_data/'
        yfiles = os.listdir(ypath)
        y = []
        for file in yfiles:
            if file.find(filename+'__result_data.csv')>=0:
                wd = pd.read_csv(ypath+file)
                wd = np.array(wd)
                print(ypath+file)

                break

        x_data = []
        y_time = []
        files.sort()
        for n in range(len(files)):
            for j in range(len(wd)):
                #print(wd[j,1])
                if files[n].find('0'+str(wd[j,1]-10)+'.npy')>0:
                    #print(files[n])
                    #print(wd[j,1])
                    x_data.append(files[n])
                    y_data.append(wd[j,3])
                    y_time.append(wd[j,0][:8])



        if len(x_data) != len(y_data):
            print(file)
            print('dismatching file')
            continue

        print(len(x_data))

        # x_data + x_data(connect)
        print(len(y_data))

        #tmp_label ={x_data[i] : y_data[i] for i in range(len(y_data)) }
        #result_y_data = np.concatenate([result_y_data,y_data])
        tmp_labels = {x_data[i] : y_data[i] for i in range(len(y_data)) }

        labels.update(tmp_labels)
        training_data = training_data + x_data
        len(training_data)
        len(labels)



    return training_data, labels,y_time, y_data


#머신러닝용 pp 데이터 출
def get_ml_pp_data(filename,flag=4):

    print(filename)

    phase_file = pd.read_excel("/home/projects/pcg_transform/PYS/phase.xlsx")
    phase_data = np.array(phase_file)

    liver_out = -1
    reperfusion = -1
    for i in range(len(phase_data)):
        if str(phase_data[i, 0]) == filename[:6]:
            liver_out = phase_data[i, 1]
            reperfusion = phase_data[i, 2]

    import os
    ypath = '/home/projects/pcg_transform/pcg_AI/ml/Data/'+'feature_data/'
    yfiles = os.listdir(ypath)
    y = []
    for file in yfiles:
        if file.find(filename+'__result_data.csv')>=0:
            wd = pd.read_csv(ypath+file)
            wd = np.array(wd)
            print(ypath+file)

            break

    txy = wd



    st = 0
    tmp = -1
    for i in range(len(txy)):
        if txy[i, 0] < str(liver_out):
                continue

        elif txy[i, 0] < str(reperfusion):
            if flag == 0:
                tmp = i
                break
        else:
            if flag == 1:
                tmp = i
                break
            elif flag == 2:
                st = i
                break


    #수정되면 한칸 밀림
    x_data = wd[st:tmp, 1:6]
    y_data = wd[st:tmp:,11]


    return x_data, y_data

#머신러닝용 pp 데이터 출
def get_ml_pp_all(filename):
    import os
    ypath = '/home/projects/pcg_transform/pcg_AI/ml/Data/'+'feature_data/'
    yfiles = os.listdir(ypath)
    y = []
    for file in yfiles:
        if file.find(filename+'_S12_total_data.csv')>=0:
            wd = pd.read_csv(ypath+file)
            wd.dropna()
            wd = np.array(wd)
            print(ypath+file)

            break


    x_data = wd[:, 1:4]
    y_data = wd[:,4]



    s12_mean = np.mean(wd[:,1])
    s12_std = np.std(wd[:,1])

    s12_stddcut = s12_mean-s12_std*2
    s12_stducut = s12_mean + s12_std * 2



    S1amp_mean = np.mean(wd[:, 2])
    S1amp_std = np.std(wd[:, 2])

    S1amp_stddcut = S1amp_mean - S1amp_std * 3
    S1amp_stducut = S1amp_mean + S1amp_std * 3



    S2amp_mean = np.mean(wd[:, 3])
    S2amp_std = np.std(wd[:, 3])

    S2amp_stddcut = S2amp_mean - S2amp_std * 3
    S2amp_stducut = S2amp_mean + S2amp_std * 3


    pp_mean = np.mean(y_data)
    pp_std = np.std(y_data)

    pp_stddcut = pp_mean - pp_std * 3
    pp_stducut = pp_mean + pp_std * 3

    i=0
    while(i<len(x_data)):
        print(i,len(x_data))
        if x_data[i,0]<s12_stddcut or x_data[i,0]>s12_stducut:
            x_data = np.delete(x_data,i,0)
            y_data = np.delete(y_data,i,0)
            continue
        if x_data[i,1]<S1amp_stddcut or x_data[i,1]>S1amp_stducut:
            x_data = np.delete(x_data,i,0)
            y_data = np.delete(y_data,i,0)
            continue
        if x_data[i,2]<S2amp_stddcut or x_data[i,2]>S2amp_stducut:
            x_data = np.delete(x_data,i,0)
            y_data = np.delete(y_data,i,0)
            continue
        if y_data[i]<pp_stddcut or y_data[i]>pp_stducut:
            x_data = np.delete(x_data,i,0)
            y_data = np.delete(y_data,i,0)
            continue

        i = i+1




    return x_data, y_data



#머신러닝용 pp 데이터 출
def get_ml_sv_data(filename):
    import os
    ypath = '/home/projects/pcg_transform/pcg_AI/ml/Data/'+'feature_data/'
    yfiles = os.listdir(ypath)
    y = []
    for file in yfiles:
        #print(file)
        if file.find(filename+'__result_data.csv')>=0:
            wd = pd.read_csv(ypath+file)
            wd = np.array(wd)
            print(ypath+file)

            break

    #수정되면 한칸 밀림
    x_data = wd[:, 1:7]
    y_data = wd[:,13]

    time = wd[:,0]


    result_x_data = []
    sv=[]

    tmp1,tmp2,tmp3,tmp4,tmp5,tmp6 = [],[],[],[],[],[]

    for i in range(10,len(time)):
        for j in range(10):
            gap = int(time[i][3:5])-int(time[i-j][3:5])
            if gap<=10 and gap>=9:
            #if gap <= 10 :
                tmp1.append(x_data[i-j,0])
                tmp2.append(x_data[i - j, 1])
                tmp3.append(x_data[i - j, 2])
                tmp4.append(x_data[i - j, 3])
                tmp5.append(x_data[i - j, 4])
                tmp6.append(x_data[i - j, 5])
        if len(tmp1) <1 :
            continue

        result_x_data.append([np.average(tmp1),np.average(tmp2),np.average(tmp3),np.average(tmp4),
                              np.average(tmp5),np.average(tmp6)])

        tmp1, tmp2, tmp3, tmp4, tmp5, tmp6 = [], [], [], [], [], []
        sv.append(y_data[i])



    return np.array(result_x_data), np.array(sv)



def next_power_of_2(x):
    """
    Find power of 2 greater than x
    """
    return 2 ** math.ceil(math.log(x) / math.log(2))

def band_pass(data, srate, fl, fh):
    """
    band pass filter
    """
    if fl > fh:
        return band_pass(data, srate, fh, fl)

    oldlen = len(data)
    newlen = next_power_of_2(oldlen)

    # srate / nsamp = Frequency increment
    # (0 ~ nsamp-1) * srate / nsamp = frequency range
    y = np.fft.fft(data, newlen)

    # filtering
    half = math.ceil(newlen / 2)
    for i in range(half):
        f = i * srate / newlen
        if f < fl or f > fh:
            y[i] = y[newlen - 1 - i] = 0

    # inverse transform
    return np.real(np.fft.ifft(y)[:oldlen])

def bland_altman_plot(data1, data2, *args, **kwargs):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')

def bland_altman_plot_20seg(data1, data2, *args, **kwargs):

    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    diff = data1 - data2  # Difference between data1 and data2
    mean = np.mean([data1, data2], axis=0)
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')

    #color = np.random.rand(20)
    for i in range(20):

        diff = data1[i::20] - data2[i::20]  # Difference between data1 and data2
        mean = np.mean([data1[i::20], data2[i::20]], axis=0)

        plt.scatter(mean, diff, *args, **kwargs,alpha=0.5,s=3)

def bland_altman_result(data1, data2, *args, **kwargs):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2

    return mean,diff