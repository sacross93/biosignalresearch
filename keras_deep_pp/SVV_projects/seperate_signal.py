from utils.my_classes import DataGenerator
import numpy as np
import matplotlib.pyplot as plt
from utils.models import *
import csv
from SVV_projects.read_SVV import read_abp_svv, read_abp_svv_10sec,read_abp_svv_minmax_fft


import pymysql
import os


stdate = 190213
enddate = 190231

signalfilepath = '/mnt/Data/abpsvv_paper_signal/'

conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                       db='abp_svv_generator', charset='utf8')
curs = conn.cursor()

sql = """select date,room_name,time,file_name,EV_SVV from abp_svv_generator.abp_svv_dataset where date >=%s and date <=%s order by date,file_name;"""
curs.execute(sql,(stdate,enddate))
row = curs.fetchall()


conn.close()

#os.mkdir(signalfilepath)

signalfilepath = '/mnt/Data/abpsvv_paper_signal/'


for i in range(len(row)):

    data = np.load(row[i][3])

    data = data[:,0].reshape([len(data), 1])
    #data.shape


    old_file_name = row[i][3].split('/')
    old_file_name = old_file_name[-1]



    save = signalfilepath + old_file_name
    # print(zfftdata.shape)
    np.savez(save, data)

    inputdata =list(row[i])
    inputdata[3] = save

    conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                           db='abp_svv_generator', charset='utf8')
    curs = conn.cursor()

    # sql = """insert into abp_svv(date,room_name,file_name,EV_SVV) values ('180712', 'D-04', filename, 11.3) ;"""

    # Connection으로부터 Cursor 생성
    # sql = """SELECT dt,PP,SBP,DBP,HR,CVP,SVV_EV from sv_trend_model.prediction_result as pr LEFT JOIN sv_trend_model.preprocessed_file as pf ON pr.id_preprocessed_file = pf.id;"""
    sql = """insert into abp_svv_dataset_signal(date,room_name,time,file_name,EV_SVV) values (%s,%s,%s,%s,%s);"""
    curs.execute(sql, (inputdata))
    #row = curs.fetchall()
    conn.commit()
    conn.close()

