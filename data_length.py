import pymysql
import numpy as np
import pandas as pd
import datetime


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


def convert_seconds_to_time(in_seconds):
    """초를 입력받아 n days, nn:nn:nn으로 변환"""
    return str(datetime.timedelta(seconds=in_seconds))

mysql_db = pymysql.connect(
    user='wlsdud1512',
    passwd='wlsdud1512',
    host='192.168.44.106',
    db='sa_server',
    charset='utf8'
)
cursor = mysql_db.cursor(pymysql.cursors.DictCursor)

#search info
search_room_name = 'PICU%'
search_begin_date = '2017-03-01'
search_end_date = '2020-12-31'
search_vital_name = ['ABP']
address="/home/projects/pcg_transform/jy/ABP_data/"

#bed_id
sql="select id from sa_api_bed where name like %s;"
cursor.execute(sql,search_room_name)
room_id = cursor.fetchall()
print(room_id)

#room_id[0]['id']

result = 0

#data_dt_search
for i in range(len(room_id)) :
    sql="select begin_date,end_date from sa_api_filerecorded where bed_id = %s"
    cursor.execute(sql,room_id[i]['id'])
    data_time = cursor.fetchall()


    for j in range(len(data_time)) :
        if data_time[j]['begin_date'] == None or data_time[j]['end_date'] == None :
            continue

        begin_stamp=datetime.datetime.timestamp(data_time[j]['begin_date'])
        end_stamp = datetime.datetime.timestamp(data_time[j]['end_date'])

        cal_time=end_stamp-begin_stamp

        if cal_time > 86400 :
            continue

        if begin_stamp != 0 and end_stamp != 0 and cal_time > 0 :
            result += cal_time
        else :
            print(data_time[j]['begin_date'],data_time[j]['end_date'])
            print("error",cal_time)

        begin_stamp = 0
        end_stamp = 0
        cal_time = 0

print(result)
24*60*60
result/60/60

print(convert_seconds_to_kor_time(result))


