import os
import numpy as np
import pandas as pd
import datetime
import pymysql
import matplotlib.pyplot as plt


stdate = 180201
#enddate = 190212
enddate = 190231

conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                       db='abp_svv_generator', charset='utf8')
curs = conn.cursor()

sql = """select date,room_name from abp_svv_generator.abp_svv_dataset where date >=%s and date <=%s group by date,room_name;"""
curs.execute(sql,(stdate,enddate))
row = curs.fetchall()

paper_set = row
len(row)


sql = """select date,room_name from abp_svv_generator.svv_test1 where date >=%s and date <=%s group by date,room_name;"""
curs.execute(sql,(stdate,enddate))
row = curs.fetchall()

no_arrhy = row
len(row)

conn.close()

len(paper_set)
len(row)

arrhy = []
for paper_data in paper_set:
    if int(paper_data[0]) > 190101:
        continue
    flag = 0
    for data in no_arrhy:
        #print(data[0],data[1])
        if paper_data[0] == data[0] and paper_data[1] == data[1]:
            flag=1
            break
    if flag ==0:
        print(paper_data)
        arrhy.append(paper_data)
import os
stpath = '/mnt/Data/CloudStation/'
result_path = '/home/projects/SVV/arrhy_datas/'


for dr in arrhy:
    os.system('cp -r '+stpath+    dr[1] + '/'+dr[0] + ' ' + result_path)
    print(dr)

