import jyLibrary as jy
import vitaldb as vr
import pandas as pd
import numpy as np

result_trks = []

roomname = [
            "D-01","D-03","D-04","D-05","D-06",
            "F-08","J-04",
            "IPACU-01","IPACU-02",
            "E-09","E-02","E-01","E-06","E-07"
            'C-02',"C-03",
            'PICU1-01','PICU1-02','PICU1-02','PICU1-04','PICU1-05','PICU1-06',
            'PICU1-07','PICU1-08','PICU1-09','PICU1-11','PICU1-10',
            'NREC-14','OB-02'
            ]

room_file = jy.searchDateRoom('D-01',21,3,8)

for i in roomname :
    room_file = jy.searchDateRoom(i,21,3,8)
    if len(room_file) != 0 :
        for k in room_file :
            trks = vr.vital_trks(k)
            for j in trks :
                if any(j in word for word in result_trks) :
                    continue
                else :
                    result_trks.append(j)
    print(len(result_trks))

df1 = pd.DataFrame(result_trks)
df1.to_excel('/home/projects/pcg_transform/jy/test3.xlsx')

len(result_trks)


