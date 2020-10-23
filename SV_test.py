import jyLibrary as jy
import datetime
import matplotlib.pyplot as plt
import numpy as np
import vr_reader_fix as vr

svDB,cursor=jy.dbIn()
room_name="D-01"
a = jy.searchDateRoom(room_name,20,8)
adrress="/mnt/data_generator/abpsvjy/"

for i in range(len(a)) :
    # abpTime,abpData = jy.findMachineInfo(a[i],None,"IBP1")
    # svTime,svData = jy.findMachineInfo(a[i],None,"SV")
    vrFile=vr.VitalFile(a[i])
    abpTime,abpData=vrFile.get_samples("IBP1")
    svTime,svData=vrFile.get_numbers("SV",)

    if len(svTime) == 0 or len(abpTime) == 0 :
        print("length issue")
        continue

    svTime=jy.timeChange(svTime,lateTime=9)
    abpTime=jy.timeChange(abpTime,"UTC")

    if len(svTime) >= 90 :
        startTime = jy.timeBinarySearch(svTime,datetime.datetime.timestamp(svTime[0]+datetime.timedelta(minutes=30)))
        endTime = jy.timeBinarySearch(svTime,datetime.datetime.timestamp(svTime[-1]-datetime.timedelta(minutes=30)))
        print("svTime length is ",len(svTime))
        svTime=svTime[startTime:endTime]
        svData=svData[startTime:endTime]
        print("cut svTime length is ",len(svTime))
        if len(svTime) == 0 :
            print("length isuue")
            continue
        startTime=jy.timeBinarySearch(abpTime,datetime.datetime.timestamp(svTime[0]))
        endTime=jy.timeBinarySearch(abpTime,datetime.datetime.timestamp(svTime[-1]))
        print("abpTime length is ",len(abpTime))
        abpTime=abpTime[startTime:endTime]
        abpData=abpData[startTime:endTime]
        print("cut abpTime length is ",len(abpTime))
        if len(abpTime) == 0 :
            print("length isuue")
            continue

    tempabp=np.where(abpData<=30)
    abpData=np.delete(abpData,tempabp)
    abpTime=np.delete(abpTime,tempabp)
    tempabp=np.where(abpData>300)
    abpData=np.delete(abpData,tempabp)
    abpTime=np.delete(abpTime,tempabp)

    std=jy.statistics(abpData,"std")
    print("")

    if std < 8 or std > 45 :
        print("std over std : ", std)
    else :

        tempCOunt = 0

        while True:
            if tempCOunt >= len(svTime) :
                break
            tempsc=0
            tempsc = jy.timeBinarySearch(svTime,datetime.datetime.timestamp(svTime[tempCOunt] + datetime.timedelta(seconds=10)))
            endAbp = jy.timeBinarySearch(abpTime,datetime.datetime.timestamp(svTime[tempCOunt] + datetime.timedelta(seconds=10)))
            startAbp = jy.timeBinarySearch(abpTime, datetime.datetime.timestamp(svTime[tempCOunt]))

            if startAbp != -1 and endAbp != -1 and tempsc != -1 :
                if abpTime[0].month < 10 and abpTime[0].day < 10:
                    cal = str(abpTime[0].year) + "0" + str(abpTime[0].month) + "0" + str(abpTime[0].day)
                elif abpTime[0].month < 10 and i.day > 10:
                    cal = str(abpTime[0].year) + "0" + str(abpTime[0].month) + str(abpTime[0].day)
                elif abpTime[0].month > 10 and i.day < 10:
                    cal = str(abpTime[0].year) + str(abpTime[0].month) + "0" + str(abpTime[0].day)
                else:
                    cal = str(abpTime[0].year) + str(abpTime[0].month) + str(abpTime[0].day)
                name = cal + "_abpdata_" + str(tempsc) + ".npz"

                # tempInt=int(svData[tempsc])
                # tempStr=str(svTime[tempsc].hour)+":"+str(svTime[tempsc].minute)+":"+str(svTime[tempsc].second)
                # tempAbpData=abpData[startAbp:endAbp]
                # tempAbpTime=abpTime[startAbp:endAbp]
                # np.savez(adrress + name, data=tempAbpData, time=tempAbpTime)
                # tempSql = "insert into abp_sv_test2(svDate,svTime,file_name,sv_value) values(%s, %s, %s, %s)"
                # tempDataSet = (cal,tempStr,adrress+name,tempInt)
                # cursor.execute(tempSql, tempDataSet)
                # svDB.commit()
                tempCOunt = tempsc
            else:
                tempCOunt += 1


            if tempsc + 6 > len(svTime):
                break

            print(tempCOunt)
            print(tempsc)







