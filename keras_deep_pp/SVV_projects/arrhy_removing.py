import os
import numpy as np
import pandas as pd
import datetime
import pymysql
import matplotlib.pyplot as plt


path = '/home/projects/SVV/arrhy_datas/abnormal_signal.csv'
wd = pd.read_csv(path)
wd = np.array(wd)

for data in wd:
    conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                           db='abp_svv_generator2', charset='utf8')
    curs = conn.cursor()

    sql = """delete from abp_svv_generator2.abp_sv_android where date =%s and room_name =%s;"""
    curs.execute(sql,(data[0],data[1]))
    conn.commit()
    conn.close()

#get arrhy matching and + arrhy
arrhy = [('190127','D-05'),('181122','D-01'),('181220','D-06'),('181207','D-04'),
         ('190102','D-06'),('181210','D-01')]
for data in arrhy:
    print(data)
    data[0]
    data[1]
    conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                           db='abp_svv_generator2', charset='utf8')
    curs = conn.cursor()

    sql = """delete from abp_svv_generator2.abp_sv_android where date =%s and room_name =%s;"""
    curs.execute(sql,(data[0],data[1]))
    conn.commit()
    conn.close()

