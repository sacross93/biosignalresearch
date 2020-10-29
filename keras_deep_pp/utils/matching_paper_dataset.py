import pymysql
import numpy as np
import csv
import pandas as pd
import os
from datetime import datetime

#load abnormal data
csvfile= '/home/projects/SVV/abnormal_signal.csv'
wd = pd.read_csv(csvfile,encoding='euc-kr')
wd = np.array(wd)
abnormal_set = wd


csvfile= '/home/projects/SVV/result_matching_svv_v2.csv'


wd = pd.read_csv(csvfile,encoding='euc-kr')
wd = np.array(wd[['수술일자','수술방','수술시작시간','수술종료시간']])



date = wd[0,0]
date = date[2:4]+date[5:7]+date[8:10]

st = wd[0,2]
et = wd[0,3]
room_name = wd[0,1]


for i in range(len(wd)):

    date = wd[i, 0]
    date = date[2:4] + date[5:7] + date[8:10]

    room = wd[i, 1]
    room = room[0] + '-' + room[1:]

    st = wd[i, 2]
    et = wd[i, 3]

    #chekc abnormal
    abnormal_flag = 0
    for abnormal in abnormal_set:
        if date == abnormal[0] and room == abnormal[1]:
            abnormal_flag=1

    if abnormal_flag == 1:
        print(date,"abnormal_data")
        continue

    print(date,'start')

    abnormal_flag = 0

    conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                           db='abp_svv_generator', charset='utf8')
    curs = conn.cursor()
    sql = """insert into abp_svv_generator.abp_svv_dataset_backup (date,room_name,time,file_name,EV_SVV)     select date,room_name,time,file_name,EV_SVV from abp_svv_generator.abp_svv_dataset where date =%s and room_name = %s and time>%s and time<%s;"""
    curs.execute(sql,  (date,room, st,et))
    row = curs.fetchall()
    conn.commit()
    conn.close()


