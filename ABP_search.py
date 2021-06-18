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
search_room_name = 'H-02'
search_begin_date = '2017-03-01'
search_end_date = '2020-12-31'
search_vital_name = ['ABP']
address="/home/projects/pcg_transform/jy/ABP_data/"

#bed_id
sql="select * from sa_api_bed where name = %s;"
cursor.execute(sql,search_room_name)
room_name = cursor.fetchall()
print(room_name)

#operation info load
rosette,tb_begin,tb_end = read_rosette_data()

print(len(rosette))

sql = "select * from sa_api_filerecorded where bed_id = %s and begin_date >= %s and end_date is not null order by begin_date asc; "
cursor.execute(sql, (room_name[0]['id'],search_begin_date))
filerecorded_result = cursor.fetchall()

print(len(filerecorded_result))

b = datetime.datetime(2020,3,1)
b = datetime.datetime.timestamp(b)

search_begin_date-=9*3600
search_end_date-=9*3600

sql = "select * from sa_api_filerecorded where bed_id = %s and ( (begin_date >= %s and begin_date <= %s and end_date is not null) or (end_date >= %s and end_date <= %s and end_date is not null ) ) order by begin_date asc; "
cursor.execute(sql, (room_name[0]['id'],search_begin_date,search_end_date,search_begin_date,search_end_date))
filerecorded_result = cursor.fetchall()

db_begin_sec = np.array([datetime.datetime.timestamp(i['begin_date']) for i in filerecorded_result])+9*3600
db_end_sec = np.array([datetime.datetime.timestamp(i['end_date']) for i in filerecorded_result])+9*3600

a = pd.to_datetime(tb_begin, format='%y-%m-%d %H:%M') # excel
tb_begin_sec = np.array([datetime.datetime.timestamp(i) for i in a.to_pydatetime()])
a = pd.to_datetime(tb_end, format='%y-%m-%d %H:%M') # excel
tb_end_sec = np.array([datetime.datetime.timestamp(i) for i in a.to_pydatetime()])

operation_result=[]
for i,j,k in zip(rosette,tb_begin_sec,tb_end_sec) :
    if i == 'F08' and j >= b:
        operation_result.append([i, j, k])

print(len(operation_result))
print(datetime.datetime.fromtimestamp(operation_result[0][1]))
print(datetime.datetime.fromtimestamp(operation_result[-1][1])+datetime.timedelta(minutes=10))

all_result=[]

for i in operation_result :
    result=[filerecorded_result[j]['file_basename'] for j in range(len(filerecorded_result)) if filerecorded_result[j]['bed_id'] == room_name[0]['id'] and (db_begin_sec[j] <= i[1] and db_end_sec[j] <= i[2] + 1800 and db_end_sec[j] >= i[2] - 1800 ) ]
    if len(result) != 0 :
        all_result.append(result)

print(len(all_result))
# def multi_create_npz(all_address) :
#     search_vr_folder = jy.searchDateRoom(search_room_name,int(all_address[0][5:7]),int(all_address[0][7:9]),int(all_address[0][9:11]))
#     operation_vr_addr = [j for j in search_vr_folder if all_address[0] in j]
#     operation_vr_file = vr.VitalFile(operation_vr_addr[0])
#
#     ABP_time,ABP_data = operation_vr_file.get_samples("ABP")
#
#     if ABP_time is None :
#         print("bad data")
#     else :
#         npz_name = operation_vr_addr[0][35:53]
#         np.savez(address + npz_name, ABP_data=ABP_data, ABP_time=ABP_time)
#
#     with Pool(4) as p :
#         p.map(multi_create_npz,all_result)


for i in range(len(all_result)) :
    search_vr_folder = jy.searchDateRoom(search_room_name,int(all_result[i][0][5:7]),int(all_result[i][0][7:9]),int(all_result[i][0][9:11]))
    operation_vr_addr = [j for j in search_vr_folder if all_result[i][0] in j]
    if len(operation_vr_addr) == 0 :
        print("no file")
        continue
    operation_vr_file = vr.VitalFile(operation_vr_addr[0])

    ABP_time,ABP_data = operation_vr_file.get_samples("IBP1")

    if ABP_time is None :
        print("None data1")
        ABP_time, ABP_data = operation_vr_file.get_samples("IBP3")

    if ABP_time is None :
        print("None data2")
        ABP_time, ABP_data = operation_vr_file.get_samples("ABP")

    if ABP_time is None :
        print("None data3")
        continue
    else :
        print("pass data")

    npz_name = operation_vr_addr[0][35:53]
    np.savez(address + npz_name, ABP_data=ABP_data,ABP_time=ABP_time)

search_vr_folder = jy.searchDateRoom(search_room_name,int(all_result[i][0][5:7]),int(all_result[i][0][7:9]),int(all_result[i][0][9:11]))
