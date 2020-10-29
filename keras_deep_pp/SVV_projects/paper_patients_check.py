import os
import numpy as np
import pandas as pd
import datetime
import pymysql
import matplotlib.pyplot as plt

path = '/home/projects/SVV/'
matching2 = 'result_matching_svv_v2.csv'
svvs = 'SVV_data_191015.xlsx'
os.path.isfile(path)

wd = pd.read_csv(path + matching2,encoding='cp949')
wd.keys()
wd = wd[['수술방','수술일자']]
wd = np.array(wd)

wd2 = pd.read_excel(path + svvs)

wd2 = wd2[['수술시작시간','수술종료시간']]
wd2 = np.array(wd2)
import dateutil.parser
import csv

with open(path + 'result_paper_patients_time.csv', 'w', newline='') as csvfile:
    wr = csv.writer(csvfile)
    #wr.writerow(['seperated set'])
    wr.writerow(['optime(<400, 400~800, >800)'])


cnt = 0
result_gap = []
low,mid,high = [],[],[]
for i in range(len(wd)):

    date = wd[i, 1]
    date = date[2:4] + date[5:7] + date[8:10]
    st = wd2[i, 0]
    st = dateutil.parser.parse(str(st))
    et = wd2[i, 1]
    et = dateutil.parser.parse(str(et))
    room = wd[i, 0]
    room = room[0] + '-' + room[1:]



    print(date,st,et,room,'start')






    conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                           db='abp_svv_generator', charset='utf8')
    curs = conn.cursor()

    sql = """select count(*) from abp_svv_generator.abp_svv_ori3c where date =%s and room_name = %s and time >=%s and time<=%s;"""
    curs.execute(sql,(date,room,st,et))
    row = curs.fetchall()

    paper_set = row
    print(row)
    row = np.array(row)
    int(row)

    conn.close()

    # result = ''
    # if int(date) <180732:
    #     result = 'training_set'
    # elif int(date) < 181232:
    #     result = 'validation_set'
    # else:
    #     result = 'test_set'
    #
    if int(row) ==0:
        result = 'problems'
        continue


    gap = (et - st).seconds // 60
    result_gap.append(gap)



    result = ''
    if gap <400:
        result = 1
        low.append(cnt)
    elif  gap <800:
        result = 2
        mid.append(cnt)
    else:
        result = 3
        high.append(cnt)

    cnt = cnt+1


    print(date)
    with open(path + 'result_paper_patients_time.csv', 'a',newline='') as csvfile:
        wr = csv.writer(csvfile)
        wr.writerow([result])



result_gap =np.array(result_gap)
np.median(result_gap)

np.percentile(result_gap,25)
np.percentile(result_gap,75)


result_gap = np.array(result_gap)
low = np.array(low)
print(
np.sum(result_gap[low]),
np.sum(result_gap[mid]),
np.sum(result_gap[high])
)
a = np.sum(result_gap[low])
b = np.sum(result_gap[mid])
c = np.sum(result_gap[high])
total = a+b+c
a/total
b/total
c/total
len(low)+len(mid)+len(high)

182/557


fig = plt.figure(figsize=(20, 10))
plt.scatter(np.arange(0,len(result_gap),1),result_gap)
plt.axhline(400, color='black', ls='--', linewidth=1)
plt.axhline(800, color='black', ls='--', linewidth=1)
plt.ylabel('duration')
plt.savefig('/home/projects/SVV/scatter.png')
#plt.show()