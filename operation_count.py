import jyLibrary as jy
import pymysql
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import vr_reader_fix as vr


def convert_seconds_to_kor_time(in_seconds):
    """초를 입력받아 읽기쉬운 한국 시간으로 변환"""
    t1 = datetime.timedelta(seconds=in_seconds)
    days = t1.days
    _sec = t1.seconds
    (hours, minutes, seconds) = str(datetime.timedelta(seconds=_sec)).split(':')
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)

    result = []
    if days >= 1:
        result.append(str(days) + '일')
    if hours >= 1:
        result.append(str(hours) + '시간')
    if minutes >= 1:
        result.append(str(minutes) + '분')
    if seconds >= 1:
        result.append(str(seconds) + '초')
    return ' '.join(result)

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

# sql="select * from sa_api_filerecorded where end_date is not null;"
# cursor.execute(sql)
# file_reecorded = cursor.fetchall()
# print(len(file_reecorded))
#
# db_begin_sec = np.array([datetime.datetime.timestamp(i['begin_date']) for i in file_reecorded])+9*3600
# db_end_sec = np.array([datetime.datetime.timestamp(i['end_date']) for i in file_reecorded])+9*3600

rosette,tb_begin,tb_end = read_rosette_data()

a = pd.to_datetime(tb_begin, format='%y-%m-%d %H:%M') # excel
tb_begin_sec = np.array([datetime.datetime.timestamp(i) for i in a.to_pydatetime()])
a = pd.to_datetime(tb_end, format='%y-%m-%d %H:%M') # excel
tb_end_sec = np.array([datetime.datetime.timestamp(i) for i in a.to_pydatetime()])

operation_count=0
file_count=0
time_length=0

for i,j,k in zip(rosette,tb_begin_sec,tb_end_sec) :

    begin_time = datetime.datetime.fromtimestamp(float(j)) - datetime.timedelta(hours=9)
    end_time = datetime.datetime.fromtimestamp(float(k)) - datetime.timedelta(hours=9)

    if begin_time.year <= 2016 :
        # print("no year")
        continue
    elif begin_time.year == 2017 and begin_time.month <= 2:
        # print("no day")
        continue

    temp_room = ''
    temp_room = i[0]+'-'+i[1:]
    # print(temp_room)

    sql="select * from sa_api_bed where name = %s;"
    cursor.execute(sql,temp_room)
    room_name = cursor.fetchall()

    if len(room_name) == 0 :
        # print("no search room")
        continue
    else :
        room_id = room_name[0]['id']

    sql="select * from sa_api_filerecorded where end_date is not null and bed_id = %s and ((begin_date >= %s and begin_date <= %s) or (end_date >= %s and end_date <= %s));"
    cursor.execute(sql,(room_id,begin_time,end_time,begin_time,end_time))
    operation_file = cursor.fetchall()
    if len(operation_file) == 0 :
        # print("no search operation file")
        continue
    else :
        print(temp_room,begin_time,end_time)
        print("search operation file")
        operation_count += 1
        file_count += len(operation_file)
        time_length += (end_time-begin_time).seconds


operation_count
file_count
result_time=convert_seconds_to_kor_time(time_length)
result_time
time_length

minute = 994983840/60
hour = minute/ 60
hour