import pymysql
import numpy as np
import csv
import pandas as pd
import os


#데이터를 문박사님 엑셀 기준으로 정해서
#모든 곳에 적용할 수 없음. 수정이 푤이




def save_abp_svv_10sec(filepath, file):
    data = []
    real_data = []
    y_data = []
    sv_data = []

    wd = pd.read_csv(filepath + file)
    tmp_y_data = np.array(wd[['EV_SVV']])
    SV_data = np.array(wd[['EV_SV']])
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

        abp_data = abp_data.reshape([len(abp_data), 1])
        data.append(abp_data.flatten())
        y_data.append(tmp_y_data[i])
        sv_data

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

    return data, np.array(y_data).flatten()


def save_svv_to_db(filepath, savepath, filename):
    conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                           db='abp_svv_generator', charset='utf8')
    data, svv = save_abp_svv_10sec(filepath, filename)
    curs = conn.cursor()

    for i in range(len(data)):
        save = savepath + filename[:-4] + '_' + str(i).zfill(6) + '.npy'

        np.save(save, data[i])

        # sql = """insert into abp_svv(date,room_name,file_name,EV_SVV) values ('180712', 'D-04', filename, 11.3) ;"""

        # Connection으로부터 Cursor 생성
        # sql = """SELECT dt,PP,SBP,DBP,HR,CVP,SVV_EV from sv_trend_model.prediction_result as pr LEFT JOIN sv_trend_model.preprocessed_file as pf ON pr.id_preprocessed_file = pf.id;"""
        sql = """insert into exist_sv(date,room_name,file_name,EV_SVV,VG_SV) values (%s,%s,%s,%s);"""
        curs.execute(sql, (filename[5:11], filename[:4], save, float(round(svv[i], 5))))

        # row = curs.fetchall()

    conn.commit()
    conn.close()



filepath = '/home/projects/pcg_transform/pcg_AI/ml/Data/Preprocessed/VG/'
filenames = os.listdir(filepath)



savepath = '/home/projects/pcg_transform/pcg_AI/deep/SVV/Data/'

import pymysql










for filename in filenames:
    save_svv_to_db(filepath,savepath,filename)
    #filename = filenames[0]
    import datetime

