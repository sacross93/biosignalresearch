import pymysql
import numpy as np
import csv
import pandas as pd
import os
from datetime import datetime

#데이터를 문박사님 엑셀 기준으로 진행




def save_abp_svv_10sec(filepath, file):
    data = []
    real_data = []
    y_data = []
    time = []

    wd = pd.read_csv(filepath + file)
    tmp_y_data = np.array(wd[['EV_SVV']])
    sv_time = np.array(wd['dt'])
    wd = np.array(wd)
    print(filepath + file)


    if len(wd) == 0:
        return -1

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

        if np.std(abp_data) < 8 or np.std(abp_data)>45:
            continue


        abp_data = abp_data.reshape([len(abp_data), 1])
        data.append(abp_data.flatten())
        y_data.append(tmp_y_data[i])
        time.append(sv_time[i])

    fft_data = []
    dpdt = []
    zfft_data = []
    for j in range(len(data)):
        # print(j)
        tmp_dpdt = np.diff(data[j].flatten(), 1)
        tmp_dpdt = np.append(tmp_dpdt, 0)
        tmp_dpdt.flatten()
        dpdt.append(tmp_dpdt)



        zfft = np.abs(np.fft.fft(data[j], 4096))
        zfft[0] = 0
        zfft = np.fft.fftshift(zfft)
        zfft = zfft[len(zfft)//2-500:len(zfft)//2+500]

        zfft_data.append(zfft)


    zfft_data = np.array(zfft_data)
    zfft_data = np.reshape(zfft_data, [len(zfft_data), 1000, 1])

    dpdt = np.array(dpdt)
    dpdt = np.reshape(dpdt, [len(dpdt), 1000, 1])

    data = np.array(data)
    data = np.reshape(data, [len(data), 1000, 1])

    data = np.dstack([data,dpdt, zfft_data])
    # data = np.dstack([data,dpdt])

    return data, np.array(y_data).flatten(),time




def save_svv_to_db(filepath, savepath, filename,type):

    conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                           db='abp_svv_generator2', charset='utf8')


    data, svv,dt = save_abp_svv_10sec(filepath, filename)
    curs = conn.cursor()

    for i in range(len(data)):
        save = savepath + filename[:-4] + '_' + str(i).zfill(6) + '.npy'
        #print(zfftdata.shape)
        np.save(save, data[i])

        # sql = """insert into abp_svv(date,room_name,file_name,EV_SVV) values ('180712', 'D-04', filename, 11.3) ;"""

        # Connection으로부터 Cursor 생성
        # sql = """SELECT dt,PP,SBP,DBP,HR,CVP,SVV_EV from sv_trend_model.prediction_result as pr LEFT JOIN sv_trend_model.preprocessed_file as pf ON pr.id_preprocessed_file = pf.id;"""
        sql = """insert into abp_svv_tableset(date,time,room_name,file_name,EV_SVV,type) values (%s,%s,%s,%s,%s,%s);"""
        curs.execute(sql, (filename[5:11],datetime.utcfromtimestamp(dt[i]+60*60*9).strftime("%y-%m-%d %H:%M:%S"), filename[:4], save, float(round(svv[i], 5)),type))




        # row = curs.fetchall()

    conn.commit()
    conn.close()


def product_helper(args):
    return save_svv_to_db(*args)

filepath = '/home/projects/pcg_transform/pcg_AI/ml/Data/Preprocessed/VG/'
filenames = os.listdir(filepath)

type = '3ch_dataset_forward'


savepath = '/mnt/Data/abpsvv_paper_tableset/'
if not os.path.isdir(savepath ):
    os.mkdir(savepath )



savepath = '/mnt/Data/abpsvv_paper_tableset/'+type+'/'
if not os.path.isdir(savepath ):
    os.mkdir(savepath )


import pymysql
filenames.sort()



conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                       db='abp_svv_generator2', charset='utf8')

curs = conn.cursor()
sql = """select date,room_name from svv_test1 group by date,room_name;"""
curs.execute(sql )
row = curs.fetchall()
row = np.array(row)

conn.close()

datas = []
for file in filenames:
    for db in row:



        if file[3] == '_' or (file[:4] == db[1] and file[5:11] == db[0]) or int(file.split('_')[1][:6])>190100:
            datas.append(file)
            break


filenames = datas

from contextlib import closing
from multiprocessing import Pool
from functools import partial
dataset = [(filepath,savepath,filenames[i],type) for i in range(len(filenames))]

with closing(Pool(15)) as pool:
    #pool.map(save_sv_to_db,[filepath,savepath,filenames])
    pool.map(product_helper,dataset  )