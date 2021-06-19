import pymysql
import numpy as np
import pandas as pd
import os
import multiprocessing
from multiprocessing import *


address = '/home/projects/pcg_transform/Meeting/jy/sv_npz/'

mysql_db = pymysql.connect(
    user='wlsdud1512',
    passwd='wlsdud1512',
    host='192.168.44.106',
    db='sa_server',
    charset='utf8'
)
cursor = mysql_db.cursor(pymysql.cursors.DictCursor)

raw_data = pd.read_excel('sv_vital_ver2.xlsx',engine='openpyxl')

for i in range(len(raw_data)) :
    temp_sv=[]
    temp_sv_time=[]
    if raw_data['ABP_count'][i] >= 500 :
        sql="""select sv,dt from number_eev where sv is not null and record_id = (select id from sa_api_filerecorded where file_basename = %s);"""
        cursor.execute(sql, raw_data['Vital_name'][i])
        sv_data = cursor.fetchall()
        for j in sv_data :
            temp_sv.append(j['sv'])
            temp_sv_time.append(j['dt'])

        np.savez(address + raw_data['Vital_name'][i][:-6],  SV_data=temp_sv, SV_time=temp_sv_time)
