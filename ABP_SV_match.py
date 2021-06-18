import pymysql
import numpy as np
import pandas as pd
import os
import multiprocessing
from multiprocessing import *

def separate_dir(dir) :

    result= dir[:-4]+".vital"

    return result


#------------------------------------------------------------define_end-----------------------------------------------------------------------------------------


mysql_db = pymysql.connect(
    user='wlsdud1512',
    passwd='wlsdud1512',
    host='192.168.44.106',
    db='sa_server',
    charset='utf8'
)
cursor = mysql_db.cursor(pymysql.cursors.DictCursor)

path_dir = os.listdir('/home/projects/pcg_transform/Meeting/jy/abp_zip/abp_unzip')

mcpu=cpu_count()

with Pool(mcpu//2) as p :
    vital_dir = p.map(separate_dir,path_dir)

vital_id_list=[]
for i in vital_dir :
    sql="""select id from sa_api_filerecorded where file_basename in (%s);"""
    cursor.execute(sql,i)
    vital_id = cursor.fetchall()

    print(vital_id)

    vital_id_list.append(vital_id)

len(vital_id_list)

raw_data=[]
for i in vital_id_list :
    try:
        vital_excel=i[0]['id']
        raw_data.append(vital_excel)
    except:
        print("error")

raw_excle={'vital_id':raw_data}
pd_excel=pd.DataFrame(raw_excle)
pd_excel.to_excel(excel_writer='vital_id.xlsx')



