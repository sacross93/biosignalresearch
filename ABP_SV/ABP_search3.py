import jyLibrary as jy
import numpy as np
import pymysql
import datetime
import vr_reader_fix as vr
from os.path import getsize

total_operation_data = np.load('/home/jmkim/.jupyter/DB_search/total_operation_data.npy',allow_pickle=True)
data_fraction = np.load('/home/jmkim/.jupyter/DB_search/data_fraction.npy', allow_pickle=True)
index_list = np.load('/home/jmkim/.jupyter/DB_search/index_list.npy', allow_pickle=True)
db_info = np.load('/home/jmkim/.jupyter/DB_search/db_info.npy', allow_pickle=True) # 1:id 2,3:begin,end(timestamp) 4:bed_id 5:path

mysql_db = pymysql.connect(
    user='wlsdud1512',
    passwd='wlsdud1512',
    host='192.168.44.106',
    db='sa_server',
    charset='utf8'
)
cursor = mysql_db.cursor(pymysql.cursors.DictCursor)

total_operation_data[0][0] # bed_name
total_operation_data[1][0] # bed_id
total_operation_data[2][0] # msecond
total_operation_data[3][0] # msecond
total_operation_data[4][0] # second begin
total_operation_data[5][0] # second end

count = 0
temp=[]
for i in range(len(data_fraction[5])) :
    if data_fraction[5][i] >= 70 and data_fraction[0][i] >= 2017 and data_fraction[0][i] < 2018 :
        count += 1
        temp.append(i)

address="/home/projects/pcg_transform/jy/new_ABP_data/"

for i in temp :

    if len(db_info[4][index_list[i][2]]) == 1 and db_info[4][index_list[i][2][0]][17:-6] != 'D-01_700101_090000':
        addr=""
        addr="/mnt/Data/CloudStation/"+db_info[4][index_list[i][2][0]][5:]
        try:
            vrfile=vr.VitalFile(addr)

            ABP_time, ABP_data = vrfile.get_samples("IBP1")

            if ABP_time is None:
                print("None data1")
                ABP_time, ABP_data = vrfile.get_samples("IBP3")

            if ABP_time is None:
                print("None data2")
                ABP_time, ABP_data = vrfile.get_samples("ABP")

            if ABP_time is None:
                print("None data3")
                ABP_time, ABP_data = vrfile.get_samples("IBP5")
            else:
                print("pass data")

            npz_name = db_info[4][index_list[i][2][0]][17:-6]
            print(npz_name)
            np.savez(address + npz_name, ABP_data=ABP_data, ABP_time=ABP_time)

        except :
            print("error")


