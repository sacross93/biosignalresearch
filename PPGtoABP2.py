import pymysql
import numpy as np
import pandas as pd
import os
import multiprocessing
from multiprocessing import *
import vr_reader_fix as vr

def separate_dir(dir) :

    result= dir[:-4]+".vital"

    return result


#------------------------------------------------------------define_end-----------------------------------------------------------------------------------------


mysql_db = pymysql.connect(
    user='wlsdud1512',
    passwd='wlsdud1512',
    host='192.168.44.106',
    db='sa_server',
    charset='utf8'
)
cursor = mysql_db.cursor(pymysql.cursors.DictCursor)

path_dir = os.listdir('/home/projects/pcg_transform/Meeting/jy/abp_zip/abp_unzip')

mcpu=cpu_count()

with Pool(mcpu//2) as p :
    vital_dir = p.map(separate_dir,path_dir)


address="/home/projects/pcg_transform/Meeting/jy/PPG_npz/"
rosetteexcel=[]
start_timeexcel=[]
end_timeexcel=[]
ABP_length=[]
PPG_length=[]

for i in vital_dir[2596:] :
    sql = """select * from sa_api_waveinfofile where (channel_name = 'PLETH' or channel_name = 'IBP1' or channel_name = 'ABP') and record_id = (select id from sa_api_filerecorded where file_basename = %s )"""
    try :
        if cursor.execute(sql, i) > 1:
            temp_abp=cursor.fetchall()
            vital_file=vr.VitalFile("/mnt/Data/CloudStation/"+i[:4]+'/'+i[5:11]+'/'+i)
            PPG_time, PPG_data=vital_file.get_samples("PLETH")
            for j in temp_abp :
                if j == 'ABP' or 'IBP1' :
                    ABP_time, ABP_data = vital_file.get_samples(j['channel_name'])
            # break
            npz_name=i[:-6]
            if PPG_time[0]>=ABP_time[0] :
                start_time=PPG_time[0]
            else :
                start_time=ABP_time[0]

            if PPG_time[-1]>=ABP_time[-1] :
                end_time=ABP_time[-1]
            else :
                end_time=PPG_time[-1]

            info={'rosette':i[:4],'start_time':start_time,'end_time':end_time,'ABP_length':len(ABP_time),'PPG_length':len(PPG_time)}
            rosetteexcel.append(i[:4])
            start_timeexcel.append(start_time)
            end_timeexcel.append(end_time)
            ABP_length.append(len(ABP_time))
            PPG_length.append(len(PPG_time))
            raw_excle = {'Rosette':rosetteexcel, 'Start_Time':start_timeexcel, 'End_Time':end_timeexcel,'ABP_length':ABP_length, 'PPG_lengh':PPG_length}
            pd_excel = pd.DataFrame(raw_excle)
            pd_excel.to_excel(excel_writer='PPG_info2.xlsx')
            np.savez(address + npz_name, info=info ,ABP_data=ABP_data, ABP_time=ABP_time, PPG_data=PPG_data, PPG_time=PPG_time)
            print(info)
    except:
        continue
    else:
        continue
