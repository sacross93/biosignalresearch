import jyLibrary as jy
import pymysql
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import vr_reader_fix as vr

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
search_room_name = 'F-08'
search_begin_date = '2020-03-01'
search_end_date = '2020-06-15'
search_vital_name = ['ECG_II','PLETH','ABP']

#bed_id
sql="select * from sa_api_bed where name = %s;"
cursor.execute(sql,search_room_name)
room_name = cursor.fetchall()
print(room_name)

#operation info load
rosette,tb_begin,tb_end = read_rosette_data()


sql = "select * from sa_api_filerecorded where bed_id = %s and begin_date >= %s and end_date is not null order by begin_date asc; "
cursor.execute(sql, (room_name[0]['id'],search_begin_date))
filerecorded_result = cursor.fetchall()

print(len(filerecorded_result))

b = datetime.datetime(2020,3,1)
b = datetime.datetime.timestamp(b)

db_begin_sec = np.array([datetime.datetime.timestamp(i['begin_date']) for i in filerecorded_result])+9*3600
db_end_sec = np.array([datetime.datetime.timestamp(i['end_date']) for i in filerecorded_result])+9*3600
a = pd.to_datetime(tb_begin, format='%y-%m-%d %H:%M') # excel
tb_begin_sec = np.array([datetime.datetime.timestamp(i) for i in a.to_pydatetime()])
a = pd.to_datetime(tb_end, format='%y-%m-%d %H:%M') # excel
tb_end_sec = np.array([datetime.datetime.timestamp(i) for i in a.to_pydatetime()])

operation_result=[]
for i,j,k in zip(rosette,tb_begin_sec,tb_end_sec) :
    if i == 'F08' and j >= b:
        operation_result.append([i,j,k])

print(len(operation_result))
print(datetime.datetime.fromtimestamp(operation_result[0][1]))
print(datetime.datetime.fromtimestamp(operation_result[-1][1])+datetime.timedelta(minutes=10))

all_result=[]

for i in operation_result :
    result=[filerecorded_result[j]['file_basename'] for j in range(len(filerecorded_result)) if filerecorded_result[j]['bed_id'] == 43 and (db_begin_sec[j] <= i[1] and db_end_sec[j] <= i[2] + 1200 and db_end_sec[j] >= i[2] - 1200 ) ]
    if len(result) != 0 :
        all_result.append(result)

# search_vr_folder = jy.searchDateRoom(search_room_name, int(all_result[0][0][5:7]), int(all_result[0][0][7:9]), int(all_result[0][0][9:11]))
# operation_vr_addr = [j for j in search_vr_folder if all_result[0][0] in j]
# operation_vr_file = vr.VitalFile(operation_vr_addr[0])
# ECG_time,ECG_data = operation_vr_file.get_samples("ECG_II")
# ABP_time,ABP_data = operation_vr_file.get_samples("ABP")
# PLETH_time,PLETH_data = operation_vr_file.get_samples("PLETH")

for i in range(len(all_result)) :
    search_vr_folder = jy.searchDateRoom(search_room_name,int(all_result[i][0][5:7]),int(all_result[i][0][7:9]),int(all_result[i][0][9:11]))
    operation_vr_addr = [j for j in search_vr_folder if all_result[i][0] in j]
    operation_vr_file = vr.VitalFile(operation_vr_addr[0])

    ECG_time,ECG_data = operation_vr_file.get_samples("ECG_II")
    ABP_time,ABP_data = operation_vr_file.get_samples("ABP")
    PLETH_time,PLETH_data = operation_vr_file.get_samples("PLETH")

    if ECG_time is None or ABP_time is None or PLETH_time is None :
        print("None data")
        continue
    else :
        print("pass data")

    npz_name = operation_vr_addr[0][35:53]
    np.savez("/home/projects/pcg_transform/jy/PTT_exam/" + npz_name, ECG_time=ECG_time,ECG_data=ECG_data,ABP_data=ABP_data,ABP_time=ABP_time,PLETH_time=PLETH_time,PLETH_data=PLETH_data)
    # tempAbpData=abpData[startAbp:endAbp]
    # tempAbpTime=abpTime[startAbp:endAbp]
    # np.savez(adrress + name, data=tempAbpData, time=tempAbpTime)


# all_waveinfo=[]
# for i in operation_result :
#     result=[ filerecorded_result[j]['id'] for j in range(len(filerecorded_result)) if (db_begin_sec[j] >= i[1] and db_begin_sec[j] <= i[2]) or (db_end_sec[j] >= i[1] and db_end_sec[j] <= i[2]) ]
#     if len(result) != 0 :
#         all_result.append(result)
#         day_waveinfo = []
#         for j in range(len(result)) :
#             sql="select * from sa_api_waveinfofile where channel_name in (%s,%s,%s) and record_id = %s;"
#             cursor.execute(sql,(search_vital_name[0],search_vital_name[1],search_vital_name[2],result[j]))
#             waveinfo_result = cursor.fetchall()
#
#             if len(waveinfo_result) == 3 :
#                 day_waveinfo.append(waveinfo_result)
#         if len(day_waveinfo) != 0 :
#             all_waveinfo.append(day_waveinfo)
#
# print(len(all_result))
# print(len(all_waveinfo))
#
#
#
# ECG = np.load("/mnt/waveNpz/"+all_waveinfo[0][0][1]['file'][10:])
#
# for k in ECG.iterkeys():
#     print(k)
#
# print(len(ECG['timestamp']))
# print(len(ECG['packet_pointer']))
# print(len(ECG['val']))

# (ECG['timestamp'][1]-ECG['timestamp'][0])/128
# ECG['timestamp'][0] + 128 * 0.002
# temp_time = np.arange(ECG['timestamp'][0],ECG['timestamp'][1],0.0020000003278255463)
# ECG_timestamp=np.array([])

#waveinfofile 쓸수 없음
# for i in range(len(ECG['packet_pointer']) -1 ) :
#     if  ECG['packet_pointer'][i+1] - ECG['packet_pointer'][i] == 128 :
#         temp_sec = (ECG['timestamp'][i+1]-ECG['timestamp'][i])/128
#         # temp_time = [ECG['timestamp'][i] + 128 * temp_sec * j for j in range(128)]
#         if temp_sec != 0 :
#             temp_time = np.arange(ECG['timestamp'][i],ECG['timestamp'][i+1],temp_sec)
#             ECG_timestamp = np.append(ECG_timestamp, temp_time)
#     else :
#         print("pointer ?")
#         print(ECG['packet_pointer'][i])
#         break
#
# print(len(ECG_timestamp))





# 수술 1개당 파일 1개 ..
# 도중에 잘린 것 처리 ?


# sql="select * from sa_api_filerecorded where bed_id = %s and ((begin_date >= %s and begin_date <= %s) or (end_date >= %s and end_date <= %s))order by begin_date asc; "
# cursor.execute(sql,(room_name[0]['id'],i['begin_date'],i['end_date'],i['begin_date'],i['end_date']))
# filerecorded_result = cursor.fetchall()