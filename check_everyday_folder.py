# -*- coding: euc-kr -*-
import os
import csv
import datetime


#현재 시간을 입력받음.
now = str(datetime.datetime.now())
today = now[2:4] + now[5:7] + now[8:10]




#현재 위치 경로 파악
syspath = '/volume1/CloudStation/CloudStation'


#######################
#checkdata today 파일이 있으면 읽어오고 시작
#result가 O인 것들 열하고 파일명 따로 저장.
#







#######################

#현재 사용중인 방들
roomname = [
            'B-01','B-02','B-03','B-04',
            'C-01','C-02','C-03','C-04','C-05','C-06',
            'D-01','D-02','D-03','D-04','D-05','D-06',
            'E-09','E-02',
            "F-01","F-02","F-03","F-04","F-05","F-07","F-08","F-09","F-10",
            "G-01","G-02","G-03","G-04","G-05","G-06",
            "H-01","H-02","H-03","H-04","H-05","H-06","H-07","H-08","H-09",
            "I-01","I-02","I-03",'I-04',
            "J-01","J-02","J-03","J-04","J-05","J-06",
            "K-01","K-02","K-04","K-05","K-06",
            'Y-01',
            "IPACU-01","IPACU-02","ER-01",
            'WREC-12','NREC-14','EREC-03'
            ]

cflag = True

readdate=["room","result"]
#checkdata 폴더에 파일 작성

f = open(syspath+"/" + "check_folder/"+ today + "_vital_check.csv","w")
wr = csv.writer(f)
wr.writerow(["room","result"])

for room in roomname:
    if os.path.isdir(syspath + "/"+room+"/"+today):
        wr.writerow([room,'O'])
        print(room + ": O")
    else:
        wr.writerow([room,'X'])
        print(room + ": X")
