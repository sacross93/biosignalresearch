import pymysql
import numpy as np
import csv
import pandas as pd
import os
from datetime import datetime

#데이터를 문박사님 엑셀 기준으로 정해서
#모든 곳에 적용할 수 없음. 수정이 푤이





def save_abp_sv_10sec(filepath, file):
    data = []
    y_data = []
    time=[]

    wd = pd.read_csv(filepath + file)
    tmp_y_data = np.array(wd[['VG_SV']])
    sv_time = np.array(wd['dt'])
    wd = np.array(wd)
    print(filepath + file)


    if len(wd) == 0:
        return -1

    # 테스트 1000개 데이터,(10초)
    # len(wd[0,-1024:-24])
    dpdt = []
    zfft_data = []
    for i in range(900, len(wd) - 900):



        #print('processing')
        abp_data = wd[i, -1024:-24]

        if np.min(abp_data[0:200]) < 30 or np.max(abp_data[0:200]) > 300 \
                or np.min(abp_data[-200:]) < 30 or np.max(abp_data[-200:]) > 300 \
                or tmp_y_data[i] < 1:
            print('range error',i ,np.min(abp_data[0:200]), np.max(abp_data[0:200]),str(tmp_y_data[i]))
            continue

        if np.std(abp_data) < 8 or np.std(abp_data)>45:
            print('std problem', i )
            continue


        abp_data = abp_data.reshape([len(abp_data), 1])
        data.append(abp_data.flatten())
        y_data.append(tmp_y_data[i])
        time.append(sv_time[i])



        # print(j)
        tmp_dpdt = np.diff(abp_data.flatten(), 1)
        tmp_dpdt = np.append(tmp_dpdt, 0)
        tmp_dpdt.flatten()
        dpdt.append(tmp_dpdt)

        zfft = np.abs(np.fft.fft(abp_data, 4096))
        zfft = np.fft.fftshift(zfft)
        zfft = zfft[len(zfft)//2-500:len(zfft)//2+500]

        zfft_data.append(zfft)



    zfft_data = np.array(zfft_data,dtype='float16')
    zfft_data = np.reshape(zfft_data, [len(zfft_data), 1000, 1])

    dpdt = np.array(dpdt,dtype='float16')
    dpdt = np.reshape(dpdt, [len(dpdt), 1000, 1])

    data = np.array(data,dtype='float16')
    data = np.reshape(data, [len(data), 1000, 1])

    data = np.dstack([data, dpdt, zfft_data])

    return data, np.array(y_data).flatten(),time



def save_sv_to_db(filepath, savepath,filename ):
    conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                           db='abp_svv_generator', charset='utf8')
    data, sv,dt = save_abp_sv_10sec(filepath, filename)
    curs = conn.cursor()
    print(len(data),len(sv))

    for i in range(len(data)):
        save = savepath+ filename[:-4] + '_' + str(i).zfill(6) + '.npy'

        np.save(save, data[i])

        sql = """insert into abp_sv(date,time,room_name,file_name,VG_SV) values (%s,%s,%s,%s,%s);"""
        curs.execute(sql, (filename[5:11],datetime.utcfromtimestamp(dt[i]).time(), filename[:4], save, float(round(sv[i], 5))))




        # row = curs.fetchall()

    conn.commit()
    conn.close()




def product_helper(args):
    return save_sv_to_db(*args)

filepath = '/home/projects/pcg_transform/pcg_AI/ml/Data/Preprocessed/VGSV/'
filenames = os.listdir(filepath)



savepath = '/mnt/Data/abpsvv/vgsv3/'
if not os.path.isdir(savepath ):
    os.mkdir(savepath )
import pymysql





filenames.sort()


from contextlib import closing
from multiprocessing import Pool
from functools import partial
"""
for filename in filenames:
    save_sv_to_db(filepath,savepath,filename)
    #filename = filenames[0]
    import datetime
"""

dataset = [(filepath,savepath,filenames[i]) for i in range(len(filenames))]

with closing(Pool(10)) as pool:
    #pool.map(save_sv_to_db,[filepath,savepath,filenames])
    pool.map(product_helper,dataset  )