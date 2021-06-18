# -*- coding: euc-kr -*-
import os
import sys
import re
import vital_reader as vt
import csv
import datetime

#날짜를 받아서 날짜폴더 파일이 있으면 True, 없으면 false
def search(today, dirname):

    flag = False
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
#        print(full_filename)
        if full_filename == (dirname +"/"+ today):
            flag = True
    return flag

#checkdata 파일 체크
def checkfile(fname):
    flag = False
    dirname = syspath+"/" + "checkdata"
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        if full_filename == (dirname + "/"+ fname):
            flag = True
    return flag

#현재 시간을 입력받음.
#now = str(datetime.datetime.now())
#today = now[2:4] + now[5:7] + now[8:10]

today = sys.argv[1]

p = re.compile("\d{6}")
if p.match(today): pass
else : raise NotImplementedError("how to use : python check_vital_pleth.py 170816")



#현재 위치 경로 파악
syspath = os.getcwd()


#######################
#checkdata today 파일이 있으면 읽어오고 시작
#result가 O인 것들 열하고 파일명 따로 저장.
#







#######################

#현재 사용중인 방들
roomname = [
            "F-01","F-02","F-03","F-04","F-05","F-07","F-08","F-09","F-10",
            "G-01","G-02","G-03","G-04","G-05","G-06",
            "H-01","H-02","H-03","H-04","H-05","H-06","H-07","H-08","H-09",
            "I-01","I-02","I-03",
            "J-01","J-02","J-03","J-04","J-05","J-06",
            "K-01","K-02","K-04","K-05","K-06",
            "NRICU-01","IPACU-01","IPACU-02","ER-01",
            ]

cflag = True

readdate=["room","day", "filename","filesize", "PLETH,ECG","result"]
#checkdata 폴더에 파일 작성
if checkfile(today + "_vital_check.csv"):
    fr = open(syspath + "/" + "checkdata/" + today + "_vital_check.csv", "r")
    reader = csv.reader(fr, delimiter=',')

    comfilename = set()
    for row in reader:
        if row[5] == 'O':
            readdate += row
            comfilename.add(row[2])

    fr.close()
    comfilenames = list(comfilename)

    f = open(syspath + "/" + "checkdata/" + today + "_vital_check.csv", "w",newline='')
    wr = csv.writer(f)

    length = len(readdate) // 6
    for num in range(length):
        wr.writerow(readdate[6 * (num):6 * (num + 1)])

    data = {}
    # 폴더가 있는 방의 데이터 전송을 확인해줌.
    for i in roomname:
        data[i] = search(today, syspath + "/" + i)
        if data[i] == True:
            filename = os.listdir(syspath + "/" + i + "/" + today)

            for j in filename:

                for comfile in comfilenames:
                    if comfile == j:
                        del comfilenames[comfilenames.index(j)]
                        cflag = False

                if cflag == True:
                    fullfilename = syspath + "/" + i + "/" + today + "/" + j
                    filesize = os.path.getsize(fullfilename)
                    # filesize 체크 후 check프로그램 돌림.
                    if filesize / 512.0 ** 2 > 1:
                        data[i] = vt.vital_reader(fullfilename)
                        print(fullfilename)
                        data[i].check_data()
                        if data[i].cflag == 0:
                            wr.writerow([i, today, j, filesize, "X", "X"])
                            # wr.writerow(str(today) + "날의" + i + "방의 " + j + "파일의 PLETH, ECG 데이터가 들어오지 않음." + "\n")
                        else:
                            wr.writerow([i, today, j, filesize, "O", "O"])
                cflag =True
        else:
            wr.writerow([i, today, "", "", "", "X"])
            print(i + " x")

else:
    f = open(syspath+"/" + "checkdata/"+ today + "_vital_check.csv","w")
    wr = csv.writer(f)
    wr.writerow(["room","day", "filename","filesize", "PLETH,ECG","result"])
    data = {}





    #폴더가 있는 방의 데이터 전송을 확인해줌.
    for i in roomname:
        data[i] = search(today, syspath + "/" + i)
        if data[i] == True:
            filename = os.listdir(syspath + "/" + i + "/" + today)
            for j in filename:

                fullfilename =  syspath + "/" + i + "/" + today +"/" +  j
                filesize = os.path.getsize(fullfilename)
                #filesize 체크 후 check프로그램 돌림.
                if filesize / 512.0 ** 2 > 1:
                    data[i] = vt.vital_reader(fullfilename)
                    print(fullfilename)
                    data[i].check_data()
                    if data[i].cflag == 0:
                        wr.writerow([i , today ,j, filesize, "X","X" ])
                        #wr.writerow(str(today) + "날의" + i + "방의 " + j + "파일의 PLETH, ECG 데이터가 들어오지 않음." + "\n")
                    else:
                        wr.writerow([i, today, j, filesize, "O", "O"])




        else:
            wr.writerow([i, today, "", "", "", "X"])
            print(i + " x")

f.close()

print("complete")
