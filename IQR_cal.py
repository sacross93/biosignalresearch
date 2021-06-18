import os
import jyLibrary as jy
import math
import numpy as np
import pymysql
import datetime
import matplotlib.pyplot as plt
from time import localtime, strftime


def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

full_file=jy.searchDateRoom("PICU1-10",20)

storage=0
recovery_count=0
for i in full_file :
    storage+=os.path.getsize(i)

print(storage)
storage_gb=convert_size(storage)
print(storage_gb)

count=0
every_file=[]
for i in range(1,13) :
    for j in range(1,32) :
        day_file = jy.searchDateRoom("PICU1-10",20,i,j)
        if len(day_file) != 0 :
            temp=0
            for k in day_file :
                temp+= os.path.getsize(k)
            every_file.append(temp)
            count += 1

print(count)
print(len(every_file))

storage_mean=storage//count
storage_mean_gb=convert_size(storage_mean)

Q1=np.percentile(every_file,0)
Q3=np.percentile(every_file,75)
IQR=Q3-Q1
IQR=convert_size(IQR)
print(IQR)
print(storage_mean_gb)

np.std(every_file)
np.var(every_file)
np.mean(every_file)
print(convert_size(np.std(every_file)))

v = sum((every_file - np.mean(every_file)) **2) / len(every_file)
print(v)
print(convert_size(v))
print(convert_size(np.var(every_file)))


plt.figure(figsize=(20, 10))
plt.plot(ART_time[length:length+5000],ART_data[length:length+5000])
plt.show()


mysql_db = pymysql.connect(
    user='wlsdud1512',
    passwd='wlsdud1512',
    host='192.168.44.106',
    db='sa_server',
    charset='utf8'
)
cursor = mysql_db.cursor(pymysql.cursors.DictCursor)

sql="select * from sa_api_bed where name = %s;"
cursor.execute(sql,"PICU1-10")
room_name = cursor.fetchall()
print(room_name)

sql="select begin_date,end_date from sa_api_filerecorded where bed_id=%s and begin_date >= '2020-01-01 00:00:00' and  begin_date < '2021-01-01 00:00:00' and end_date is not null order by begin_date;"
cursor.execute(sql,room_name[0]['id'])
file_name = cursor.fetchall()

(file_name[1:]['begin_date']-file_name[:-1]['end_date']).seconds

for i in range(1,len(file_name)) :
    print((file_name[i]['begin_date']-file_name[i-1]['end_date']).seconds)



""""
bed별 데이터있는 날짜 개수
날짜별 평균 기록 시간
std var <- 하는김에...
20년걸로
"""
save_recovery_data=[]
bed="PICU1-0"
for i in range(1,12) :
    if i < 10 :
        sql = "select * from sa_api_bed where name = %s;"
        cursor.execute(sql, bed + str(i))
        room_name = cursor.fetchall()
        print(room_name)

    else :
        sql = "select * from sa_api_bed where name = %s;"
        cursor.execute(sql, "PICU1-"+str(i))
        room_name = cursor.fetchall()
        print(room_name)

    sql = "select * from sa_api_filerecorded where bed_id = %s and begin_date >= '2020-01-01 00:00:00' and begin_date < '2021-01-01 00:00:00' and end_date is not null order by begin_date asc;"
    cursor.execute(sql,room_name[0]['id'])
    filerecorded = cursor.fetchall()

    temp_month=999
    temp_day=0
    record_time=0
    day_time=[]
    for j in filerecorded :
        recent_month=j['end_date'].month
        recent_day=j['end_date'].day

        if recent_day == temp_day :
            record_time += (j['end_date']-j['begin_date']).seconds
        else :
            if record_time != 0 :
                day_time.append(record_time)
                temp_day = recent_day
                record_time=0
            else :
                record_time = (j['end_date']-j['begin_date']).seconds

    print(room_name[0]['name'])
    temp_dic={'name':room_name[0]['name'],'data':day_time}
    save_recovery_data.append(temp_dic)

len(save_recovery_data[0]['data'])

mean_cal=0
mean=[]
std=[]
var=[]
mean_bed=[]
for i in save_recovery_data :
    mean_cal+=len(i['data'])
    mean_bed.append(len(i['data']))
    mean.append(int(np.mean(i['data'])//60//60))
    std.append(int(np.std(i['data'])//60//60))
    var.append(int(np.var(i['data'])))
print(mean_cal)
mean_cal=mean_cal//len(save_recovery_data)
print(mean_cal)
print(mean)
print(std)
print(var)
print(mean_bed)

np.mean(mean)
np.std(std)
np.var(var)

