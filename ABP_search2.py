import jyLibrary as jy
import pymysql
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import vr_reader_fix as vr
import multiprocessing
from multiprocessing import *

def isNaN(string):
    return string != string

def read_rosette_data():
    path = '/home/jmkim/.jupyter/DB_search/OPER_ROOM_DATE_FROM_2015.csv'
    rosette_data = pd.read_csv(path)
    rosette = np.array(rosette_data["Rosette"])
    #     rosette = (rosette_data["Rosette"])
    begin_date = np.array(pd.to_datetime(rosette_data["Start"]))
    end_date = np.array(pd.to_datetime(rosette_data["End"]))

    for i in range(1, len(rosette)):
        if isNaN(rosette[i]) == True: rosette[i] = rosette[i - 1]

    ind = np.where((pd.isnull(begin_date) == False) & (pd.isnull(end_date) == False))[0]

    return rosette[ind], begin_date[ind], end_date[ind]

mysql_db = pymysql.connect(
    user='wlsdud1512',
    passwd='wlsdud1512',
    host='192.168.44.106',
    db='sa_server',
    charset='utf8'
)
cursor = mysql_db.cursor(pymysql.cursors.DictCursor)

#search info
# D 1,2,3,4,5,6 F-2,7,8,9 G-3,4,6
search_room_name = 'D-02'
search_begin_date = '2017-03-01'
search_end_date = '2021-02-20'
search_vital_name = ['ABP']
address="/home/projects/pcg_transform/jy/ABP_data/"

#bed_id
sql="select * from sa_api_bed where name = %s;"
cursor.execute(sql,search_room_name)
room_name = cursor.fetchall()
print(room_name)
print(len(room_name))

#operation info load
rosette,tb_begin,tb_end = read_rosette_data()

print(len(rosette))

a = pd.to_datetime(tb_begin, format='%y-%m-%d %H:%M') # excel
tb_begin_sec = a.to_pydatetime()
a = pd.to_datetime(tb_end, format='%y-%m-%d %H:%M') # excel
tb_end_sec = a.to_pydatetime()

#filerecorded_search
sql = "select * from sa_api_filerecorded where bed_id= %s and ((begin_date >= %s and begin_date <= %s) or (end_date >= %s and end_date <= %s)) order by begin_date asc;"
cursor.execute(sql, (room_name[0]['id'], tb_begin_sec[0], tb_end_sec[0], tb_begin_sec[0], tb_end_sec[0]))
file_recorded = cursor.fetchall()
print(len(file_recorded))

for i in range(len(rosette)) :

    if tb_begin_sec[i].year < 2018 or (tb_begin_sec[i].year == 2018 and tb_begin_sec[i].month == 8 and tb_begin_sec[i].day == 3 and rosette[i] == 'B02' ) :
        continue

    #bed_id search
    temp_room_name=rosette[i][0]+'-'+rosette[i][1:]
    sql = "select * from sa_api_bed where name = %s;"
    cursor.execute(sql, temp_room_name)
    room_name = cursor.fetchall()

    if len(room_name) == 0 :
        continue

    print(room_name[0]['name'])
    print(tb_begin_sec[i])

    #filerecorded_search
    sql = "select * from sa_api_filerecorded where bed_id= %s and ((begin_date >= %s and begin_date <= %s) or (end_date >= %s and end_date <= %s)) and end_date is not null order by begin_date asc;"
    cursor.execute(sql, (room_name[0]['id'], tb_begin_sec[i] - datetime.timedelta(hours=9), tb_end_sec[i] - datetime.timedelta(hours=9), tb_begin_sec[i] - datetime.timedelta(hours=9), tb_end_sec[i] - datetime.timedelta(hours=9)))
    file_recorded = cursor.fetchall()

    if len(file_recorded) == 0 :
        continue

    elif len(file_recorded) == 1 :
        print(file_recorded[0]['file_basename'])
        search_vr_folder = jy.searchDateRoom(room_name[0]['name'],int(file_recorded[0]['file_basename'][5:7]),int(file_recorded[0]['file_basename'][7:9]),int(file_recorded[0]['file_basename'][9:11]))
        if len(search_vr_folder) == 1 :
            operation_vr_file = vr.VitalFile(search_vr_folder[0])
        elif len(search_vr_folder) == 0 :
            continue
        else :
            for j in search_vr_folder :
                if j.find(file_recorded[0]['file_basename']) != -1 :
                    operation_vr_file = vr.VitalFile(j)

        ABP_time, ABP_data = operation_vr_file.get_samples("IBP1")

        if ABP_time is None:
            print("None data1")
            ABP_time, ABP_data = operation_vr_file.get_samples("IBP3")

        if ABP_time is None:
            print("None data2")
            ABP_time, ABP_data = operation_vr_file.get_samples("ABP")

        if ABP_time is None:
            print("None data3")
            continue
        else:
            print("pass data")

        npz_name = file_recorded[0]['file_basename'][0:18]
        np.savez(address + npz_name, ABP_data=ABP_data, ABP_time=ABP_time)

    else :
        #for j in file_recorded :
        continue

ABP_time, ABP_data = operation_vr_file.get_samples("ABP")