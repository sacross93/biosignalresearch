import pymysql
import numpy as np
import csv
import pandas as pd
import os
import datetime

def read_signal_data(path):
    with open(path, 'rb')as f:
        content = f.read()
        x = np.array(np.frombuffer(content, dtype=np.float32))
    return x


def get_pcg_sv_data(filenames, x_path='/home/jmkim/data2',y='SV' ,delay=10):
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
            if file.find(filename) >= 0:
                files.append(file)

        files.sort()
        # x_data = sort_file(files)

        ypath = '/home/projects/pcg_transform/pcg_AI/ml/Data/feature_data/'
        yfiles = os.listdir(ypath)
        for file in yfiles:
            if file.find(filename + '__result_data.csv') >= 0:
                wd = pd.read_csv(ypath + file)
                y_data = np.array(wd[[y]])
                print(ypath + file)
                y_time = np.array(wd)[:, 0]

                break

        x_data = []
        result_data = []
        files.sort()
        #for n in range(len(files)):
        for i in range(10, len(y_time)):

            cnt = i
            while (1):
                if cnt == -1 or i-30 > cnt:
                    break

                svtime = datetime.datetime.combine(datetime.date.today(), datetime.datetime.strptime(y_time[i],"%H:%M:%S.%f").time()) - datetime.timedelta(
                        minutes=10)
                svchecktime = datetime.datetime.combine(datetime.date.today(),  datetime.datetime.strptime(y_time[cnt],"%H:%M:%S.%f").time())

                if  svtime.minute == svchecktime.minute and svtime.hour== svchecktime.hour :
                    result_data.append(y_data[cnt])
                    x_data.append(files[cnt])

                    break
                cnt = cnt - 1


        y_data = result_data

        print(len(x_data))

        # x_data + x_data(connect)
        print(len(y_data))


        if len(x_data) != len(y_data):
            print(file)
            print('dismatching file')
            continue

        print(len(x_data))

        # x_data + x_data(connect)
        print(len(y_data))

        # tmp_label ={x_data[i] : y_data[i] for i in range(len(y_data)) }
        # result_y_data = np.concatenate([result_y_data,y_data])
        tmp_labels = {x_data[i]: y_data[i] for i in range(len(y_data))}

        labels.update(tmp_labels)
        training_data = training_data + x_data
        len(training_data)
        len(labels)

    return training_data, labels, y_time, y_data



def get_pcg_svv_data(filenames, x_path='/home/jmkim/data2',y='SVV' ,delay=10):
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
            if file.find(filename) >= 0:
                files.append(file)

        files.sort()
        # x_data = sort_file(files)

        ypath = '/home/projects/pcg_transform/pcg_AI/ml/Data/feature_data/'
        yfiles = os.listdir(ypath)
        for file in yfiles:
            if file.find(filename + '__result_data.csv') >= 0:
                wd = pd.read_csv(ypath + file)
                y_data = np.array(wd[[y]])
                print(ypath + file)
                y_time = np.array(wd)[:, 0]

                break

        x_data = []
        files.sort()
        #for n in range(len(files)):
        for j in range(len(wd)):
            x_data.append(files[j])

        print(len(x_data))

        # x_data + x_data(connect)
        print(len(y_data))


        if len(x_data) != len(y_data):
            print(file)
            print('dismatching file')
            continue

        print(len(x_data))

        # x_data + x_data(connect)
        print(len(y_data))

        # tmp_label ={x_data[i] : y_data[i] for i in range(len(y_data)) }
        # result_y_data = np.concatenate([result_y_data,y_data])
        tmp_labels = {x_data[i]: y_data[i] for i in range(len(y_data))}

        labels.update(tmp_labels)
        training_data = training_data + x_data
        len(training_data)
        len(labels)

    return training_data, labels, y_time, y_data