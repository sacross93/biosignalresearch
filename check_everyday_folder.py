# -*- coding: euc-kr -*-
import os
import csv
import datetime


#���� �ð��� �Է¹���.
now = str(datetime.datetime.now())
today = now[2:4] + now[5:7] + now[8:10]




#���� ��ġ ��� �ľ�
syspath = '/volume1/CloudStation/CloudStation'


#######################
#checkdata today ������ ������ �о���� ����
#result�� O�� �͵� ���ϰ� ���ϸ� ���� ����.
#







#######################

#���� ������� ���
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
#checkdata ������ ���� �ۼ�

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
