import pymysql
import numpy as np
import csv
import pandas as pd
import os
from utils.processing import band_pass
from sklearn.preprocessing import MinMaxScaler
import datetime

def read_abp_svv_10sec(filedates):
    data = []
    real_data = []
    y_data = []
    time = []
    for filedate in filedates:
        # filepath = '/home/shmoon/Preprocessed/VG/'
        filepath = '/home/projects/pcg_transform/pcg_AI/ml/Data/Preprocessed/VG/'
        files = os.listdir(filepath)

        wd = []
        # filedate = '180202'
        for file in files:
            if file.find(filedate + '.csv') >= 0:
                wd = pd.read_csv(filepath + file)
                tmp_y_data = np.array(wd[['EV_SVV']])

                wd = np.array(wd)

                print(filepath + file)

                break

        if len(wd) == 0:
            continue

        # 테스트 1000개 데이터,(10초)
        # len(wd[0,-1024:-24])

        for i in range(900, len(wd) - 900):

            abp_data = wd[i, -1024:-24]

            if np.min(abp_data[0:200]) < 30 or np.max(abp_data[0:200]) > 300 \
                    or np.min(abp_data[-200:]) < 30 or np.max(abp_data[-200:]) > 300 \
                    or tmp_y_data[i] < 1:
                continue

            time.append((datetime.datetime.utcfromtimestamp(wd[i,0]) + datetime.timedelta(hours=9)).time())

            abp_fft = np.fft.fft(abp_data, 1000)
            abp_fft[0] = 0
            abp_data = np.fft.ifft(abp_fft)

            abp_data = abp_data.reshape([len(abp_data), 1])
            data.append(abp_data.flatten())
            y_data.append(tmp_y_data[i])

    fft_data = []
    dpdt = []
    for j in range(len(data)):
        # print(j)
        tmp_data = []

        for k in range(10):
            # print(k)
            tmp_fft = np.abs(np.fft.fft(data[j][k:k + 100], 100))
            tmp_fft[0] = 0
            tmp_fft = np.fft.fftshift(tmp_fft)
            tmp_data.append(tmp_fft)

        tmp_dpdt = np.diff(data[j].flatten(), 1)
        tmp_dpdt = np.append(tmp_dpdt, 0)
        tmp_dpdt.flatten()
        dpdt.append(tmp_dpdt)

        tmp_data = np.array(tmp_data)
        tmp_data = tmp_data.flatten()
        fft_data.append(tmp_data)

    fft_data = np.array(fft_data)
    fft_data = np.reshape(fft_data, [len(fft_data), 1000, 1])

    dpdt = np.array(dpdt)
    dpdt = np.reshape(dpdt, [len(dpdt), 1000, 1])

    data = np.array(data)
    data = np.reshape(data, [len(data), 1000, 1])

    data = np.dstack([data, dpdt, fft_data])

    print(data.shape)

    return data, np.array(y_data), np.array(time), file[:4]

def read_abp_svv_10sec_old(filedates):
    data = []
    real_data = []
    y_data = []
    for filedate in filedates:
        # filepath = '/home/shmoon/Preprocessed/VG/'
        filepath = '/home/projects/pcg_transform/pcg_AI/ml/Data/Preprocessed/VG/'
        files = os.listdir(filepath)

        wd = []
        # filedate = '180202'
        for file in files:
            if file.find(filedate + '.csv') >= 0:
                wd = pd.read_csv(filepath + file)
                tmp_y_data = np.array(wd[['EV_SVV']])
                wd = np.array(wd)
                print(filepath + file)

                break

        if len(wd) == 0:
            continue

        # 테스트 1000개 데이터,(10초)
        # len(wd[0,-1024:-24])

        for i in range(900, len(wd) - 900):

            abp_data = wd[i, -1024:-24]

            if np.min(abp_data[0:200]) < 30 or np.max(abp_data[0:200]) > 300 \
                    or np.min(abp_data[-200:]) < 30 or np.max(abp_data[-200:]) > 300 \
                    or tmp_y_data[i] < 1:
                continue

            abp_fft = np.fft.fft(abp_data, 1000)
            abp_fft[0] = 0
            abp_data = np.fft.ifft(abp_fft)

            abp_data = abp_data.reshape([len(abp_data), 1])
            data.append(abp_data.flatten())
            y_data.append(tmp_y_data[i])

    fft_data = []
    dpdt = []
    for j in range(len(data)):
        # print(j)
        tmp_data = []

        for k in range(10):
            # print(k)
            tmp_fft = np.abs(np.fft.fft(data[j][k:k + 100], 100))
            tmp_fft[0] = 0
            tmp_fft = np.fft.fftshift(tmp_fft)
            tmp_data.append(tmp_fft)

        tmp_dpdt = np.diff(data[j].flatten(), 1)
        tmp_dpdt = np.append(tmp_dpdt, 0)
        tmp_dpdt.flatten()
        dpdt.append(tmp_dpdt)

        tmp_data = np.array(tmp_data)
        tmp_data = tmp_data.flatten()
        fft_data.append(tmp_data)

    fft_data = np.array(fft_data)
    fft_data = np.reshape(fft_data, [len(fft_data), 1000, 1])

    dpdt = np.array(dpdt)
    dpdt = np.reshape(dpdt, [len(dpdt), 1000, 1])

    data = np.array(data)
    data = np.reshape(data, [len(data), 1000, 1])

    data = np.dstack([data, dpdt, fft_data])

    print(data.shape)

    return data, np.array(y_data)

def read_abp_sv_10sec(filedates):

    data = []
    real_data = []
    y_data = []
    for filedate in filedates:
        #filepath = '/home/shmoon/Preprocessed/VG/'
        filepath = '/home/projects/pcg_transform/pcg_AI/ml/Data/Preprocessed/VG/'
        files = os.listdir(filepath)

        wd  =[]
        #filedate = '180202'
        for file in files:
            if file.find(filedate + '.csv') >= 0:
                wd = pd.read_csv(filepath + file)
                tmp_y_data = np.array(wd[['EV_SV']])
                wd = np.array(wd)
                print(filepath + file)

                break

        if len(wd)==0:
            continue

        #테스트 1000개 데이터,(10초)
        #len(wd[0,-1024:-24])

        for i in range(900,len(wd)-900):


                abp_data = wd[i,-1024:]

                if np.min(abp_data) < 30 or np.max(abp_data)>300\
                        or tmp_y_data[i] <1:
                    continue
                abp_data = abp_data.reshape([len(abp_data),1])
                data.append(abp_data)
                y_data.append(tmp_y_data[i])

    fft_data = []
    dpdt = []
    for j in range(len(data)):
        #print(j)
        tmp_data = []
        for k in range(20):
            #print(k)
            tmp_fft = np.abs(np.fft.fft(real_data[j][k:k+100],100))
            #tmp_fft[0] = 0
            tmp_data.append(tmp_fft)

        tmp_dpdt = np.diff(data[j].flatten(),1)
        tmp_dpdt = np.append(tmp_dpdt, 0)
        tmp_dpdt.flatten()
        dpdt.append(tmp_dpdt)

        tmp_data = np.array(tmp_data)
        tmp_data = tmp_data.flatten()
        fft_data.append(tmp_data)


    fft_data = np.array(fft_data)
    fft_data = np.reshape(fft_data, [len(fft_data), 1024, 1])

    dpdt = np.array(dpdt)
    dpdt = np.reshape(dpdt, [len(dpdt), 1024, 1])

    data = np.array(data)
    data = np.reshape(data, [len(data), 1024, 1])

    data = np.dstack([data, dpdt, fft_data])

    print(data.shape)

    return data,np.array(y_data)

def read_abp_svv(filedates):

    data = []
    real_data = []
    y_data = []
    for filedate in filedates:
        #filepath = '/home/shmoon/Preprocessed/VG/'
        filepath = '/home/projects/pcg_transform/pcg_AI/ml/Data/Preprocessed/VG/'
        files = os.listdir(filepath)

        wd  =[]
        #filedate = '180202'
        for file in files:
            if file.find(filedate + '.csv') >= 0:
                wd = pd.read_csv(filepath + file)
                tmp_y_data = np.array(wd[['EV_SVV']])
                wd = np.array(wd)
                print(filepath + file)

                break

        if len(wd)==0:
            continue

        #테스트 1000개 데이터,(10초)
        #len(wd[0,-1024:-24])

        min_max_scaler = MinMaxScaler()
        for i in range(len(wd)-10):


            if tmp_y_data[i+5] !=0.0:
                abp_data = np.concatenate([wd[i,-1024:-24],wd[i+5,-1024:-24]])

                if np.min(abp_data) < 30 or np.max(abp_data)>300:
                    continue
                abp_data = abp_data.reshape([len(abp_data),1])
                #abp_data = band_pass(abp_data,100,1,100)
                real_data.append(abp_data.flatten())

                #abp_data = min_max_scaler.fit_transform(abp_data)
                #abp_data = abp_data.flatten()

                data.append(abp_data)
                y_data.append(tmp_y_data[i+5])

    real_data = np.array(real_data)
    fft_data = []
    dpdt = []
    for j in range(len(real_data)):
        #print(j)
        tmp_data = []
        for k in range(20):
            #print(k)
            tmp_fft = np.abs(np.fft.fft(real_data[j][k:k+100],100))
            #tmp_fft[0] = 0
            tmp_data.append(tmp_fft)

        tmp_dpdt = np.diff(real_data[j],1)
        tmp_dpdt = np.append(tmp_dpdt, 0)
        tmp_dpdt.flatten()
        dpdt.append(tmp_dpdt)

        tmp_data = np.array(tmp_data)
        tmp_data = tmp_data.flatten()
        fft_data.append(tmp_data)


    fft_data = np.array(fft_data)
    fft_data = np.reshape(fft_data, [len(fft_data), 2000, 1])

    dpdt = np.array(dpdt)
    dpdt = np.reshape(dpdt, [len(dpdt), 2000, 1])

    data = np.array(data)
    data = np.reshape(data, [len(data), 2000, 1])

    data = np.dstack([data, dpdt, fft_data])

    print(data.shape)

    return data,np.array(y_data)



def read_abp_svv_minmax_all(filedates):

    data = []
    real_data = []
    y_data = []
    for filedate in filedates:
        #filepath = '/home/shmoon/Preprocessed/VG/'
        filepath = '/home/projects/pcg_transform/pcg_AI/ml/Data/Preprocessed/VG/'
        files = os.listdir(filepath)

        wd  =[]
        #filedate = '180202'
        for file in files:
            if file.find(filedate + '.csv') >= 0:
                wd = pd.read_csv(filepath + file)
                tmp_y_data = np.array(wd[['EV_SVV']])
                wd = np.array(wd)
                print(filepath + file)

                break

        if len(wd)==0:
            continue

        #테스트 1000개 데이터,(10초)
        #len(wd[0,-1024:-24])

        min_max_scaler = MinMaxScaler()
        for i in range(len(wd)-10):


            if tmp_y_data[i+10] !=0.0:
                abp_data = np.concatenate([wd[i,-1024:-24],wd[i+5,-1024:-24]])

                if np.min(abp_data) < 30 or np.max(abp_data)>300:
                    continue
                abp_data = abp_data.reshape([len(abp_data),1])
                #abp_data = band_pass(abp_data,100,1,100)
                real_data.append(abp_data.flatten())

                abp_data = min_max_scaler.fit_transform(abp_data)
                abp_data = abp_data.flatten()

                data.append(abp_data)
                y_data.append(tmp_y_data[i+10])

    real_data = np.array(real_data)
    fft_data = []
    dpdt = []
    for j in range(len(real_data)):
        #print(j)
        tmp_data = []
        for k in range(20):
            #print(k)
            tmp_fft = np.abs(np.fft.fft(real_data[j][k:k+100],100))
            tmp_fft[0] = 0
            tmp_data.append(tmp_fft)

        tmp_dpdt = np.diff(real_data[j],1)
        tmp_dpdt = np.append(tmp_dpdt, 0)
        tmp_dpdt.flatten()
        dpdt.append(tmp_dpdt)

        tmp_data = np.array(tmp_data)
        tmp_data = tmp_data.flatten()
        fft_data.append(tmp_data)


    fft_data = np.array(fft_data)
    fft_data = np.reshape(fft_data, [len(fft_data), 2000, 1])

    dpdt = np.array(dpdt)
    dpdt = np.reshape(dpdt, [len(dpdt), 2000, 1])

    data = np.array(data)
    data = np.reshape(data, [len(data), 2000, 1])

    data = np.dstack([data, dpdt, fft_data])

    print(data.shape)

    return data,np.array(y_data)


def read_abp_svv_minmax_fft(filedates):


    data = []
    real_data = []
    y_data = []
    for filedate in filedates:
        wd = []
        #filepath = '/home/shmoon/Preprocessed/VG/'
        filepath = '/home/projects/pcg_transform/pcg_AI/ml/Data/Preprocessed/VG/'
        files = os.listdir(filepath)

        #filedate = '180202'
        for file in files:
            if file.find(filedate + '.csv') >= 0:
                wd = pd.read_csv(filepath + file)
                tmp_y_data = np.array(wd[['EV_SVV']])
                wd = np.array(wd)
                print(filepath + file)

                break

        if len(wd)==0:
            continue

        #테스트 1000개 데이터,(10초)
        #len(wd[0,-1024:-24])

        min_max_scaler = MinMaxScaler()
        for i in range(len(wd)-10):

            if tmp_y_data[i+10] !=0.0:
                abp_data = np.concatenate([wd[i,-1024:-24],wd[i+5,-1024:-24]])

                if np.min(abp_data) < 30 or np.max(abp_data)>300:
                    continue
                abp_data = abp_data.reshape([len(abp_data),1])
                #abp_data = band_pass(abp_data,100,1,100)
                #real_data.append(abp_data.flatten())

                #abp_data = min_max_scaler.fit_transform(abp_data)
                abp_data = abp_data.flatten()

                data.append(abp_data)
                y_data.append(tmp_y_data[i+10])

    #real_data = np.array(real_data)
    fft_data = []
    dpdt = []
    for j in range(len(data)):
        #print(j)
        tmp_data = []
        for k in range(20):
            #print(k)
            tmp_fft = np.abs(np.fft.fft(data[j][k:k+100],100))
            tmp_data.append(tmp_fft)

        tmp_dpdt = np.diff(data[j],1)
        tmp_dpdt = np.append(tmp_dpdt, 0)
        tmp_dpdt.flatten()
        dpdt.append(tmp_dpdt)

        tmp_data = np.array(tmp_data)
        tmp_data = tmp_data.flatten()
        fft_data.append(tmp_data)


    fft_data = np.array(fft_data)
    fft_data = np.reshape(fft_data, [len(fft_data), 2000, 1])

    dpdt = np.array(dpdt)
    dpdt = np.reshape(dpdt, [len(dpdt), 2000, 1])

    data = np.array(data)
    data = np.reshape(data, [len(data), 2000, 1])

    data = np.dstack([data, dpdt, fft_data])

    print(data.shape)

    return data,np.array(y_data)




if __name__ == '__main__':
    conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                           db ='sv_trend_model', charset='utf8')

    curs = conn.cursor()

    #Connection으로부터 Cursor 생성
    #sql = """SELECT dt,PP,SBP,DBP,HR,CVP,SVV_EV from sv_trend_model.prediction_result as pr LEFT JOIN sv_trend_model.preprocessed_file as pf ON pr.id_preprocessed_file = pf.id;"""
    sql = """SELECT dt,PP,SBP,DBP,HR,CVP,SVV_EV from sv_trend_model.prediction_result where dt like '2018-02-01%' """
    curs.execute(sql)

    row = curs.fetchall()
    print(row)

    conn.close()


def read_svv_dataset(filedate):
    conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                           db='sv_trend_model', charset='utf8')

    curs = conn.cursor()

    # Connection으로부터 Cursor 생성
    # sql = """SELECT dt,PP,SBP,DBP,HR,CVP,SVV_EV from sv_trend_model.prediction_result as pr LEFT JOIN sv_trend_model.preprocessed_file as pf ON pr.id_preprocessed_file = pf.id;"""
    sql = """SELECT dt,PP,SBP,DBP,HR,CVP,SVV_EV from sv_trend_model.prediction_result where dt like '2018-02-01%' """
    curs.execute(sql)
    row = curs.fetchall()

    conn.close()

    data = np.array(row)
    result_data = []
    result_y_data = []
    for i in range(10,len(data)):
        fdata = data[i-10:i,1:-2]
        result_data.append(fdata.flatten())
        result_y_data.append(data[i,-1])






    return np.array(result_data),np.array(result_y_data)

def read_sv_dataset(filedate):
    conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                           db='sv_trend_model', charset='utf8')

    curs = conn.cursor()

    # Connection으로부터 Cursor 생성
    # sql = """SELECT dt,PP,SBP,DBP,HR,CVP,SVV_EV from sv_trend_model.prediction_result as pr LEFT JOIN sv_trend_model.preprocessed_file as pf ON pr.id_preprocessed_file = pf.id;"""
    sql = """SELECT dt,PP,SBP,DBP,HR,CVP,SV_EV from sv_trend_model.prediction_result where dt like '2018-02-01%' """
    curs.execute(sql)
    row = curs.fetchall()

    conn.close()
    return np.array(row)