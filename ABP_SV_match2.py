import pymysql
import numpy as np
import pandas as pd
import os
import multiprocessing
from multiprocessing import *

mysql_db = pymysql.connect(
    user='wlsdud1512',
    passwd='wlsdud1512',
    host='192.168.44.106',
    db='sa_server',
    charset='utf8'
)
cursor = mysql_db.cursor(pymysql.cursors.DictCursor)

raw_data = pd.read_excel('vital_id.xlsx',engine='openpyxl')

num_packets_list=[]
sv_excel=[]
id_excel=[]
sv_count=0
for i in raw_data['vital_id'] :
    sv_id = 0
    sql = """select count(SV) from number_eev where record_id = %s;"""
    cursor.execute(sql, i)
    sv_id = cursor.fetchall()

    if sv_id[0]['count(SV)'] > 100:
        sql = """select file_basename from sa_api_filerecorded where id = %s"""
        cursor.execute(sql,i)
        vital_name=cursor.fetchall()
        sql = """select * from sa_api_waveinfofile where record_id = %s and (channel_name like 'IBP1' or channel_name like 'IBP5' or channel_name like 'ABP' ) ;"""
        cursor.execute(sql, i)
        waveinfo = cursor.fetchall()
        print(sv_id)
        sv_count += 1
        sv_excel.append(sv_id[0]['count(SV)'])
        id_excel.append(vital_name[0]['file_basename'])
        try :
            num_packets_list.append(waveinfo[0]['num_packets'])
        except :
            num_packets_list.append(0)

raw_excel={'Vital_name':id_excel,'SV_count':sv_excel,'ABP_count':num_packets_list}
pd_data=pd.DataFrame(raw_excel)
pd_data.to_excel(excel_writer='sv_vital_ver2.xlsx')



