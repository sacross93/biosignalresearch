import os
import datetime
import shutil

roomname = [
            "D-01","D-02","D-03","D-04","D-05","D-06",
            "F-01","F-02","F-03","F-04","F-05","F-06","F-07","F-08","F-09","F-10",
            "G-01","G-02","G-03","G-04","G-05","G-06",
            "H-01","H-02","H-04","H-05","H-06","H-07","H-08","H-09",
            "I-01","I-02","I-03",'I-04','Y-01',
            "J-01","J-02","J-03","J-04","J-05","J-06",
            "K-01","K-02","K-03","K-04","K-05","K-06",
            "IPACU-01","IPACU-02",
            "E-09","E-02","E-01",
            "C-01",'C-02',"C-03",'C-04','C-05','C-06',
            'B-01','B-02','B-03','B-04',
            'PICU1-01','PICU1-02','PICU1-02','PICU1-04','PICU1-05','PICU1-06',
            'PICU1-07','PICU1-08','PICU1-09','PICU1-11','PICU1-10',
            'NREC-14','OB-01','OB-02'
            



            ]

for room in roomname:
    if os.path.isdir(room):
        dir = os.listdir(room)
        for d in dir:
            if os.path.isdir(room+"/"+d):

                today = str(datetime.datetime.today())
                yesterday =  str(datetime.datetime.today()-datetime.timedelta(days=1))
                day = today[2:4] + today[5:7] + today[8:10]
                yday = yesterday[2:4] + yesterday[5:7] + yesterday[8:10]
                if d != day and d != yday:
                    print(room+"/"+d)
                    shutil.rmtree(room+"/"+d,ignore_errors=False,onerror=None)
