import os
import numpy as np
import pandas as pd
import datetime
import pymysql
import matplotlib.pyplot as plt
from utils.processing import *
import matplotlib.dates as mdates

path = '/home/projects/SVV/svv_paperset_t3/'
file_name = 'D-02_190207SVV.csv'
#
# path = '/home/projects/SVV/'
# file_name = 'D-02_181114SVV.csv'

date = 190207
room_name = 'D-02'
wd = pd.read_csv(path + file_name)
time = wd[['time']]
wd = np.array(wd)

val_time = np.array(wd[:,2],dtype='datetime64')
#datetime.datetime(val_time[0])
tmp_time = val_time.astype(datetime.datetime)
val_time = []
for dt in tmp_time:
    val_time.append(dt.time())

val_time = np.array(val_time)



pred = wd[:,3]
svv = wd[:,4]


index_min = np.argmin(wd[:,4])
index_max = np.argmax(wd[:,4])
max_time = wd[index_max,2]
min_time = wd[index_min,2]

index_max2 = 6500 + np.argmax(wd[10000:,4])
# index_max2 = 8000 + np.argmax(wd[10000:,4])
index_min2 = 7000 + np.argmin(wd[7000:8000,4])
max_time2 = wd[index_max2,2]
min_time2 = wd[index_min2,2]





def get_target_data(date,room_name,time):
    conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                           db='abp_svv_generator2', charset='utf8')
    curs = conn.cursor()

    sql = """select file_name from abp_svv_ori3c where date =%s and room_name=%s and time = %s;"""
    curs.execute(sql,(date,room_name,time))
    row = curs.fetchall()

    paper_set = row
    len(row)
    conn.close
    data = np.load(str(row[0][0]))
    return data

max_data = get_target_data(date,room_name,max_time)
min_data = get_target_data(date,room_name,min_time)
max_data2 = get_target_data(date,room_name,max_time2)
min_data2 = get_target_data(date,room_name,min_time2)



max_data.shape


timehour = []
timeidx = []
for i in range(len(val_time)-1):
    if val_time[i].hour != val_time[i+1].hour:
        timehour.append(str(val_time[i+1].hour))
        timeidx.append(i+1)

timehour = np.array(timehour)
timeidx = np.array(timeidx)

#plt.close()


fig = plt.figure(figsize=(20, 20))
plt.subplot(2,1,1)
#plt.title(room_name + ' ' + str(date))
plt.plot(val_time, pred, 'b', label="Predicted")
plt.plot(val_time, svv, 'r', label='SVV')
#if float(max(svv)) > 12.0:
#    plt.axhline(12, color='black', ls='--', linewidth=1)
plt.xticks(val_time[timeidx],timehour.flatten(),size='20')
plt.yticks(size='20')



plt.axvline(max_time, color='black', ls='--', linewidth=1)
plt.axvline(min_time, color='black', ls='--', linewidth=1)
plt.axvline(max_time2, color='black', ls='--', linewidth=1)
plt.axvline(min_time2, color='black', ls='--', linewidth=1)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# plt.xticks([i * (len(pred) // 5 - 1) for i in range(6)])
plt.legend(fontsize='xx-large',frameon=False)
plt.xlabel('hours',size='20')
plt.gca().xaxis.set_label_coords(0.97,-0.05)


plt.subplot(8,1,5)
# plt.title('a',loc='left',fontweight="bold", size = 15)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.plot(min_data[:,0],'r')
plt.ylabel('A',size=25,rotation='horizontal')
plt.yticks(size='17')
plt.xlim(-10,1000)
plt.ylim(-30, 50)

plt.xticks([])

plt.subplot(8,1,6)
# plt.title('b',loc='left',fontweight="bold", size = 15)
plt.plot(max_data[:,0],'r')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.ylim(-30, 50)
plt.ylabel('B',size=25,rotation='horizontal')
plt.yticks(size='17')
plt.xlim(-10,1000)
plt.xticks([])

plt.subplot(8,1,7)
# plt.title('c',loc='left',fontweight="bold", size = 15)
plt.plot(min_data2[:,0],'r')

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.ylim(-30, 50)
plt.ylabel('C',size=25,rotation='horizontal')
plt.yticks(size='17')
plt.xlim(-10,1000)
plt.xticks([])

plt.subplot(8,1,8)
# plt.title('d',loc='left',fontweight="bold", size = 15)
plt.ylabel('D',size=25,rotation='horizontal')
plt.plot(max_data2[:,0],'r')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.ylim(-30, 50)
plt.xticks(np.arange(0, 1100, 100), np.arange(0, 11, 1),size=20)
plt.xlim(-10,1000)
plt.xlabel('seconds',size='20')
plt.gca().xaxis.set_label_coords(0.97,-0.2)
#plt.show()
plt.yticks(size='17')

plt.savefig('/home/projects/SVV/result_figure/' + 'result_figure_200521' + '.png',bbox_inches='tight',pad_inches=1.0,dpi=300)

#plt.show()
plt.close()
