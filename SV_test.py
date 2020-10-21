import jyLibrary as jy
import datetime

a = jy.searchDateRoom("D-01",20,8,3)

abpTime = [0 for i in range(len(a))]
abpData = [0 for i in range(len(a))]
svTime = [0 for i in range(len(a))]
svData = [0 for i in range(len(a))]

for i in range(len(a)) :
    abpTime[i],abpData[i] = jy.findMachineInfo(a[i],None,"IBP1")
    svTime[i],svData[i] = jy.findMachineInfo(a[i],None,"SV")

for i in range(len(svTime)) :
    startTime = jy.timeBinarySearch(svTime[i],datetime.datetime.timestamp(svTime[i][0]+datetime.timedelta(minutes=30)))
    print("svTime length is ",len(svTime[i]))
    svTime[i]=svTime[i][startTime:]
    print("cut svTime length is ",len(svTime[i]))




