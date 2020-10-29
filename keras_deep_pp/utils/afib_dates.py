import pymysql
import numpy as np
import csv
import pandas as pd
import os
from datetime import datetime
import math



csvfile= '/home/projects/SVV/EKG_forStudy.xlsx'
savepath = '/home/projects/SVV/abnormal_signal.csv'


wd = pd.read_excel(csvfile)
wd = np.array(wd[['수술일자','로젯','Proc시작시간','Proc종료시간','수술방']])


date = wd[0,0]
date = str(date.year)[2:] + str(date.month).zfill(2) + str(date.day).zfill(2)
rossette = wd[0,1]
st = wd[0,2]
et = wd[0,3]
room_name = wd[0,4]


for i in range(449,len(wd)):
    rossette = wd[i, 1]
    if rossette !='D' or math.isnan(st) or math.isnan(et):
        print(rossette,st,et)
        continue


    date = wd[i, 0]
    date = str(date.year)[2:] + str(date.month).zfill(2) + str(date.day).zfill(2)
    st = wd[i, 2]
    et = wd[i, 3]
    room = wd[i, 4]
    room = room[0] + '-' + room[1:]


    conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                           db='abp_svv_generator', charset='utf8')
    curs = conn.cursor()
    sql = """select file_name,EV_SVV,room_name,date from abp_svv_generator.abp_svv_dataset where date =%s and room_name = %s and time>%s and time<%s;"""
    curs.execute(sql, (date,room, st,et))
    row = curs.fetchall()
    conn.close()

    if len(row) == 0:
        #print(date,'is not have data')
        continue


    print(date)
    with open(savepath, 'a',newline='') as csvfile:
        wr = csv.writer(csvfile)
        wr.writerow([date,room])




