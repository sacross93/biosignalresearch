import pymysql
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
import struct
import datetime
from sklearn.preprocessing import StandardScaler
from PCG_SVV.PCG_DB_method import *
from sklearn.preprocessing import minmax_scale


def svv_save_db(date,room_name,npdata,time,svv,compcnt,type):

    base_dir =  '/mnt/data_generator/pcgsvv_' + type + '/'
    if not os.path.isdir(base_dir ):
        os.mkdir(base_dir)
    svv_save_directory = base_dir+ room_name+'/' + str(date) +'/'

    if not os.path.isdir(base_dir + room_name + '/'):
        os.mkdir(base_dir + room_name + '/')

    if not os.path.isdir(svv_save_directory):
        os.mkdir(svv_save_directory)
    path = svv_save_directory+str(compcnt).zfill(6)+'.npz'
    np.savez(path, npdata)
    conn = pymysql.connect(host='localhost', user='jmkim', password='anesthesia',
                           db='data_generator', charset='utf8')
    curs = conn.cursor()

    # print(tSVV_time)
    # Connection으로부터 Cursor 생성

    sql = """insert into PCG_SVV (date,room_name,time,pcg_file,svv,type) values (%s,%s,%s,%s,%s,%s);"""
    curs.execute(sql, (date,room_name,time,path,svv,type))
    conn.commit()
    conn.close()

HOSTNAME = 'localhost'
USERNAME = 'jmkim'
PASSWORD = 'anesthesia'
DBNAME = 'data_generator'
DEVICE_DB_NAME = 'Vital_DB'





#xlsx_path = '/home/projects/pcg_transform/pcg_analysis/PCG_list_190822.xlsx'
#xlsx_path = '/home/projects/pcg_transform/pcg_analysis/PCG_list_191112.xlsx'
xlsx_path = '/home/projects/pcg_transform/pcg_analysis/PCG_list_191112v2.xlsx'#v2
wd = pd.read_excel(xlsx_path)
wd = np.array(wd)
#print(wd)


pcg_track_name='file_name'
room = 'D-06'
device = 'EV1000'
device_track_name = 'SVV'
type = 'minmax'
# pcg_track_name = 'wS12'
# device = 'Vigilance'
# device_track_name = 'SV'
pcgtime,pcg_file,svv = [],[],[]
for i in range(len(wd)):

    date = int(wd[i,0])
    print(date)
    # reper_time = wd[i,2]
    # ptID = wd[i,1]
    # PRS = wd[i,3]
    # levo = wd[i,4]
    # epi = wd[i,5]
    # meld = wd[i,6]

    time, pcg_track = load_pcg_feature_files_v2(room, date, pcg_track_name)
    #time, pcg_track = load_pcg_feature_files_v2(room, date, pcg_track_name)
    match_time, match_pcg, cnt,svvtime,svvdata = load_matching_device_pcg10svv(room, time, pcg_track, device,
                                                              device_track_name)

    compcnt = 0
    for i in range(len(cnt)):
        if len(cnt) // 4 == i:
            print('processing 1/4')
        match_time[i]
        for j in range(cnt[i][0],cnt[i][1]):
            #check 10sec exist
            gap = svvtime[j] - match_time[i]

            gap_range = gap.seconds * 1000 + gap.microseconds // 1000
            if gap_range > 20000:
                continue


            if svvtime[j]-match_time[i]<datetime.timedelta(seconds=10):
                if i ==0:
                    continue

                #check pre data exist
                if svvtime[j]-match_time[i-1]<datetime.timedelta(seconds=30):
                    predata = np.load(match_pcg[i-1])
                    nowdata = np.load(match_pcg[i])

                    pre = predata['arr_0'][:,-(10000-gap_range):,:]
                    after = nowdata['arr_0'][:,:gap_range:,:]

                    result = np.hstack([pre,after])
                    #print(result.shape)
                    if len(result[0,:,0]) != 10000:
                        raise Exception('3의 배수가 아닙니다.')
                else:
                    continue

            else:
                nowdata = np.load(match_pcg[i])
                result = nowdata['arr_0'][:,gap_range-10000:gap_range,:]
                #print(result.shape)
                if len(result[0, :, 0]) != 10000:
                    raise Exception('3의 배수가 아닙니다.')

            #result = np.dstack([result[:,:,0],result[:,:,3]])
            data1 = np.array(result[:,:,0],dtype=np.float16)
            data1 = data1[0]
            data2 = np.array(result[:, :, 3], dtype=np.float16)
            data2 = data2[0]

            data1= minmax_scale(data1,axis=0,copy=True)
            data2= minmax_scale(data2,axis=0,copy=True)

            # fig = plt.figure(figsize=(20, 10))
            # plt.subplot(2, 1, 1)
            # plt.plot(data1)
            # plt.subplot(2, 1, 2)
            # plt.plot(data2)
            # plt.savefig('std after')

            data1 = np.reshape(data1,[1,len(data1),1])
            data2 = np.reshape(data2,[1,len(data2),1])

            result = np.dstack([data1,data2])
            #result.shape
            svv_save_db(date, room,result,svvtime[j],svvdata[j],compcnt,type)
            compcnt = compcnt+1


    print('save_complete')



