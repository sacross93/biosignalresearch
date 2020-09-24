import vr_reader_fix as vr
import multiprocessing
import pymysql
from multiprocessing import *
import parmap
import anssignal
import check_filepath as cf
import os
import matplotlib.pyplot as plt
import datetime
import pytz
import numpy as np
import pandas
import time
import copy
import scipy.stats as stats
import pylab
import pickle

def search(dirname):
    full_filename = []
    addressTemp = ("/mnt/CloudStation/")
    if dirname.find("/mnt/CloudStation/") == -1 :
        dirname = addressTemp + dirname

    filenames = os.listdir(dirname)

    for filename in filenames:
        full_filename.append(os.path.join(dirname, filename))
    return full_filename

def searchRoomAllFile(roomname):
    roomInfo = search(roomname)
    # print(roomInfo)
    file = []
    result = []
    etc=['/mnt/Data/CloudStation/D-05/170714_080052.vital','/mnt/CloudStation/D-05/170714_080052.vital','/mnt/CloudStation/D-01/check_vital_pleth.py','/mnt/CloudStation/D-01/vital_reader.py']

    for i in etc :
        if i in roomInfo :
            roomInfo.remove(i)
    while True:
        if roomInfo is None or not roomInfo:
            break
        else:
            temp = roomInfo.pop()
            if temp in etc:
                temp = roomInfo.pop()
            temp = search(temp)
            for i in range(len(temp)):
                file.append(temp.pop())
    while True:
        if file is None or not file:
            break
        else:
            temp = file.pop()
            if os.path.getsize(temp) > 5000000:
                result.append(temp)

    result.sort()
    return result

def searchDate(roomName,year, month=None, day=None):
    vrfile=[]

    for i in roomName :
        if year < int(i[23:25]) :
            break
        if type(int(i[23:25]))== str or type(int(i[27:29])) == str or type(int(i[25:27])) == str :
            continue
        elif year == int(i[23:25]) :
            if month != None and day != None :
                if month == int(i[25:27]) and day == int(i[27:29]) :
                    vrfile.append(i)
            elif month != None and day == None :
                if month == int(i[25:27]) :
                    vrfile.append(i)
            elif month == None and day != None :
                if str(day) in int(i[27:29]) :
                    vrfile.append(i)
            else :
                vrfile.append(i)

    return vrfile

def searchWaveform(data):
    waveForm = ['ECG_II', 'ART1', 'ART2', 'VOLT', 'IBP1', 'IBP3', 'PLETH', 'ECG1', 'CO2', 'AWP','AUDIO','ABP']
    if any(data in word for word in waveForm):
        return True
    else:
        return False

def findMachineInfo(target, machineName=None, machineName2=None):
    # count = 0
    # selfData=['/mnt/CloudStation/F-08/200810/F-08_200810_080534.vital', '/mnt/CloudStation/F-08/200810/F-08_200810_104015.vital']
    resultTime = []
    resultData = []
    num_cores = multiprocessing.cpu_count()

    # parResult = parmap.map(vr.VitalFile, target, pm_pbar=True, pm_processes=5)
    with Pool(4) as p:
        parResult = p.map(vr.VitalFile, target)

    tempDic={'machineName' : machineName,'machineName2':machineName2,'vrFile':parResult }
    print(parResult)
    # print(len(parResult))
    for i in parResult:
        time, data = getData(i, machineName, machineName2)
        # print(data[0].shape)
        if time is not None or time != None :
            resultTime=np.append(resultTime,time)
            resultData=np.append(resultData,data)
    #     count+=1

    return resultTime, resultData

def multiGetData(vrDic) :
    Dtime=[]
    Ddata=[]
    print(vrDic)

def getData(vrfile, machineName=None, machineName2=None):
    # count=0
    # print(vrfile)
    # print(machineName)
    # print(machineName2)
    Dtime = []
    Ddata = []
    for trk in vrfile.trks.values():
        # count += 1
        # print(count)
        dname = vrfile.devs[trk['did']]
        if machineName is not None:
            # print("machineName not None")
            if dname['name'] == machineName:
                # print(dname['name']==machineName)
                if machineName2 is not None:
                    if trk['name'].find(machineName2):
                        if searchWaveform(machineName2):
                            Dtime, Ddata = vrfile.get_samples(machineName2, machineName)
                            break
                            # print(len(Dtime))
                            # print(len(Ddata))
                        else:
                            # print(2)
                            # print("correct")
                            Dtime, Ddata = vrfile.get_numbers(machineName2, machineName)
                            break
        else:
            if machineName2 is not None:
                if trk['name'].find(machineName2):
                    if searchWaveform(machineName2):
                        # print(len(trk['name']))
                        Dtime, Ddata = vrfile.get_samples(machineName2)
                        # print(1)
                        break
                        # print(len(Dtime))
                        # print(len(Ddata))
                    else:
                        # print(2)
                        # print("correct")
                        Dtime, Ddata = vrfile.get_numbers(machineName2)
                        break

    # print(Ddata)
    return Dtime, Ddata

def statistics(data, require=None):
    average = np.mean(data)
    std = np.std(data)
    var = np.var(data)
    # print("average= ",average," std= ",std," var= ",var)
    if require is not None:
        if require == 'average':
            return average
        if require == 'std':
            return std
        if require == 'var':
            return var
    else:
        return average, std, var

def quartile(temp):  # quartile function
    data = np.array([])
    data = np.append(data, temp)
    data.sort()
    Q1 = len(data) * 0.25
    if Q1 % 1 != 0:
        Q1 = (data[int(Q1)] + data[int(Q1) - 1]) / 2
    else:
        Q1 = data[int(Q1) - 1]
    Q2 = len(data) * 0.5
    if Q2 % 1 == 0:
        Q2 = (data[int(Q2)] + data[int(Q1) - 1]) / 2
    else:
        Q2 = data[int(Q2) + 1]
    Q3 = len(data) * 0.75
    if Q3 % 1 != 0:
        Q3 = (data[int(Q3)] + data[int(Q3) - 1]) / 2
    else:
        Q3 = data[int(Q3) - 1]
    Q4 = data[int(len(data) * 1) - 1]

    return Q1, Q2, Q3, Q4

def outLier(data):  # box plot outLier search
    Q = quartile(data)
    IQR = Q[2] - Q[0]
    result = [Q[0] - (1.5 * IQR), Q[2] + (1.5 * IQR)]
    return result

def fivePerOutLier(data) :
    maxData=max(data)*0.95
    minTemp=min(data)-min(data)*0.95
    minData=min(data)+minTemp
    return maxData,minData

def plot(time, data):
    # print("plot start")
    newData = np.array([])
    newTime = np.array([])
    newData = np.append(newData, data)
    newTime = np.append(newTime, time)
    # print("outLier in")
    Q = outLier(newData)
    # print("plot in")
    plt.figure(figsize=(20, 10))
    plt.plot(newTime, newData)
    plt.axhline(y=statistics(newData, 'average'), color='r', linewidth=1)
    # num_cores = multiprocessing.cpu_count()
    # temp=np.array_split(data,num_cores)
    # parResult = parmap.map(a, temp, pm_pbar=True, pm_processes=num_cores)
    # for i in range(len(data)):
    #     # print("for in")
    #     if Q[0] > data[i] or Q[1] < data[i]:
    #         # print("if in")
    #         plt.axvline(x=newTime[i], color='r', linestyle='--', linewidth=0.5)
    # print(Q[0],Q[1])
    plt.show()
    plt.close()

def fivePlot(time,data) :
    # print("plot start")
    newData = np.array([])
    newTime = np.array([])
    newData = np.append(newData, data)
    newTime = np.append(newTime, time)
    # print("outLier in")
    maxData,minData = fivePerOutLier(data)
    # print("plot in")
    plt.plot(newTime, newData)
    plt.axhline(y=statistics(newData, 'average'), color='r', linewidth=1)
    # num_cores = multiprocessing.cpu_count()
    # temp=np.array_split(data,num_cores)
    # parResult = parmap.map(a, temp, pm_pbar=True, pm_processes=num_cores)
    for i in range(len(data)):
        # print("for in")
        if minData > data[i] or maxData < data[i]:
            # print("if in")
            plt.axvline(x=newTime[i], color='r', linestyle='--', linewidth=0.5)
    # print(Q[0],Q[1])
    plt.show()
    plt.close()

def timeChange(time, timeType, lateTime=None):
    tempTime = []
    a=np.array([])
    if timeType == 'timestamp':
        for i in range(len(time)):
            if type(time[i]) == type(a) :
                continue
            elif lateTime is None:
                temp=datetime.datetime.timestamp(time[i])
                tempTime.append(temp)
            else:
                temp=datetime.datetime.timestamp(time[i] + datetime.timedelta(hours=lateTime))
                tempTime.append(temp)
    elif timeType == 'UTC':
        for i in range(len(time)):
            if type(time[i]) == type(a) :
                continue
            elif lateTime is None:
                tempTime.append(datetime.datetime.fromtimestamp(time[i]))
            else:
                tempTime.append(datetime.datetime.timestamp(time[i] + datetime.timedelta(hours=lateTime)))
    return tempTime

def waveMatch(time1, data1, time2, data2) :
    checkSvTime=np.array([])
    checkSvData=np.array([])
    tenIbpData=[]
    tenIbpTime=[]
    matchtime = time1[0].second % 10
    for i in range(len(time1)) :
        if time1[i].second % 10 == matchtime :
            for j in range(10) :
                minVal=-1
                temp=time1[i] - datetime.timedelta(seconds=10-j)
                temp=datetime.datetime.timestamp(temp)
                minVal=timeBinarySearch(time2,temp)
                if minVal != -1 :
                    break
            tempVal=datetime.datetime.timestamp(time1[i])
            Val=timeBinarySearch(time2,tempVal)
            checkSvTime=np.append(checkSvTime,time1[i])
            checkSvData=np.append(checkSvData,data1[i])
            tenIbpTime.append(time2[minVal:Val])
            tenIbpData.append(data2[minVal:Val])

    return checkSvTime , checkSvData , tenIbpTime , tenIbpData

def timeMatchData(time1, data1, time2, data2, btime=None):
    resultTime1 = np.array([])
    resultTime2 = np.array([])
    matchData1 = np.array([])
    matchData2 = np.array([])
    matchtime = time1[0].second % 10
    for i in range(len(time1)):
        # print("for")
        if time1[i].second % 10 == matchtime:
            # print("if")
            tempValues = timeBinarySearch(time2, time1[i])
            # print("if2")
            if tempValues is not -1:
                # print(j)
                # print(len(data2))
                # print("1")
                matchData1 = np.append(matchData1, data1[i])
                # print("2")
                resultTime1 = np.append(resultTime1, time1[i])
                # print("3")
                matchData2 = np.append(matchData2, data2[tempValues])
                # print("4")
                resultTime2 = np.append(resultTime2, time2[tempValues])
                # print("5")
    # print("out")
    return resultTime1, matchData1, resultTime2, matchData2

def timeBinarySearch(time,target) :
    low=0
    high=len(time)-1
    # if str(type(target)) == "<class 'datetime.datetime'>" :
    #     target=int(datetime.datetime.timestamp(target))
    while(low<=high) :
        # print("??")
        mid = int((low+high) / 2)
        # print(type(time[mid]))
        # print(target)
        if int(datetime.datetime.timestamp(time[mid])) > int(target) :
            # print("2")
            high = mid - 1
        elif int(datetime.datetime.timestamp(time[mid])) < int(target) :
            # print("3")
            low = mid + 1
        else :
            print(time[mid])
            print("search")
            break
    # print(mid)
    if mid == 0 or mid == len(time):
        return -1
    else :
        return mid

def timeSeach(time,data,startTime,endTime) :
    # starttime 105545 endtime 105630
    minSecond=startTime%100
    # print(minSecond)
    minMinute=((startTime-minSecond)%10000)//100
    # print(minMinute)
    minHour=(startTime-minSecond-minMinute)//10000
    # print(minHour)
    target = datetime.datetime(time[0].year,time[0].month,time[0].day, minHour, minMinute, minSecond)
    target = datetime.datetime.timestamp(target)
    minTime = timeBinarySearch(time,target)

    maxSecond=endTime%100
    maxMinute=((endTime-maxSecond)%10000)//100
    maxHour=(endTime-maxSecond-maxMinute)//10000
    target = datetime.datetime(time[0].year,time[0].month,time[0].day,maxHour,maxMinute, maxSecond)
    target = datetime.datetime.timestamp(target)
    maxTime = timeBinarySearch(time,target)

    print(minTime)
    print(maxTime)
    resultTime = np.array([])
    matchData = np.array([])
    if minTime == -1 or maxTime == -1 :
        print("search break")
        return resultTime,matchData

    matchData = np.array(data[minTime:maxTime])
    matchData=np.ravel(matchData,order='c')
    resultTime = np.array(time[minTime:maxTime])
    print(resultTime.shape)
    resultTime=np.ravel(resultTime,order='c')
    print("finish")

    return resultTime, matchData

def binaryGoodTime(time,data,beforTime,t_hour=None,t_minute=None,t_second=0) :
    for i in range(beforTime) :
        target = datetime.datetime(time[0].year,time[0].month,time[0].day, t_hour, t_minute, t_second)
        # print(type(target))
        target = datetime.datetime.timestamp(target - datetime.timedelta(minutes=beforTime + i))
        minTime = timeBinarySearch(time,target)
        if minTime != -1 :
            break
    for i in range(beforTime) :
        target = datetime.datetime(time[0].year,time[0].month,time[0].day,t_hour,t_minute, t_second)
        # print(type(target))
        target = datetime.datetime.timestamp(target + datetime.timedelta(minutes=beforTime - i ))
        maxTime = timeBinarySearch(time,target)
        if maxTime != -1 :
            break
    print(minTime)
    print(maxTime)
    resultTime = np.array([])
    matchData = np.array([])
    if minTime == -1 or maxTime == -1 :
        print("search break")
        return resultTime,matchData
    if time[minTime].hour > t_hour :
        print("hour break")
        return resultTime,matchData

    matchData = np.array(data[minTime:maxTime])
    matchData=np.ravel(matchData,order='c')
    resultTime = np.array(time[minTime:maxTime])
    print(resultTime.shape)
    resultTime=np.ravel(resultTime,order='c')
    print("finish")

    return resultTime, matchData

def saveData(data, name):
    np.savez("/home/wlsdud1512/testNpz001/" + name + ".npz", data)
    # np.savez("/home/wlsdud1512/testNpz001/testIBP0731.npz", testTime=IBP0731["0731IBPTIME"],testData=IBP0731["0731IBPDATA"])
    #
    # with np.load("/home/wlsdud1512/testNpz001/testIBP0731.npz",allow_pickle=True) as data :
    #     testT=data['testTime']

def saveDic(data,name) :
    name= "/home/wlsdud1512/testNpz001/" +name+".pickle"
    with open(name, 'wb') as fw:
        pickle.dump(data, fw)

def loadDic(name) :
    name = "/home/wlsdud1512/testNpz001/" +name+".pickle"
    with open(name, 'rb') as fr:
        result = pickle.load(fr)
    return result

def dbIn() :
    result = pymysql.connect(
        user='wlsdud1512',
        passwd='wlsdud1512',
        host='192.168.134.184',
        db='data_generator'
    )
    cursor = result.cursor(pymysql.cursors.DictCursor)

    return result , cursor

def insertDB(svtime,svdata,ibpDic) :
    db,cursor = dbIn()
    tempSql="insert into SVtest_jy(room_name,date,time,file_name,SV_value) values(%s, %s, %s, %s,%s)"
    dd, tt = splitTime(svtime)
    for i in range(len(svtime)):
        tempStr= "/home/wlsdud1512/testNpz001/"+dd[i][4:8]+"IBP"+str(i+1)+".npz"
        np.savez(tempStr, Time=ibpDic[dd[i][4:8]+"IBPTIME"][i], Data=ibpDic[dd[i][4:8]+"IBPDATA"][i])
        tempDataSet = ("D-05", dd[i], tt[i], tempStr, float(svdata[i]))
        cursor.execute(tempSql, tempDataSet)
        db.commit()

def testDb(svtime,svdata,ibpDic) :
    db, cursor = dbIn()
    tempSql = "insert into SVtest_jy3(room_name,date,time,file_name,SV_value) values(%s, %s, %s, %s,%s)"
    dd, tt = splitTime(svtime)
    for i in range(len(svtime)):
        tempStr = "/home/wlsdud1512/testNpz001/" + dd[i][4:8] + "IBP" + str(i + 1) + ".npz"
        np.savez(tempStr, Time=ibpDic["TIME"][i], Data=ibpDic["DATA"][i])
        tempDataSet = ("D-05", dd[i], tt[i], tempStr, float(svdata[i]))
        cursor.execute(tempSql, tempDataSet)
        db.commit()

def splitTime(time):
    DBT = []
    for i in time:
        if i.month < 10 and i.day < 10:
            DBT.append(str(i.year) + "0" + str(i.month) + "0" + str(i.day))
        elif i.month < 10 and i.day > 10:
            DBT.append(str(i.year) + "0" + str(i.month) + str(i.day))
        elif i.month > 10 and i.day < 10:
            DBT.append(str(i.year) + str(i.month) + "0" + str(i.day))
        else:
            DBT.append(str(i.year) + str(i.month) + str(i.day))
    DBD = []
    for i in time:
        temp = ''
        if i.hour < 10:
            temp += "0" + str(i.hour)
        else:
            temp += str(i.hour)
        if i.minute < 10:
            temp += "0" + str(i.minute)
        else:
            temp += str(i.minute)
        if i.second < 10:
            temp += "0" + str(i.second)
        else:
            temp += str(i.second)
        DBD.append(temp)

    return DBT,DBD

start = time.time()
roomData=searchRoomAllFile("D-05")
# print(len(roomData))
# print(roomData)
D05room08=searchDate(roomData,20,8)
print("loadiong...")
# a = vr.VitalFile(D01room0908[0])
# b = vr.VitalFile(D01room0908[1])

# print(time.time() - start)

print(time.time() - start)

# print(D05room0731)
# D05room0804=searchDate(roomData,20,8,4)
# D05room0805=searchDate(roomData,20,8,5)
# D05room0806=searchDate(roomData,20,8,6)
# D05room0812=searchDate(roomData,20,8,12)
#
#
# D05roomAll=searchDate(roomData,20)
# svt,svd = findMachineInfo(D05roomAll,'EV1000','SV')
# ibpt,ibpd = findMachineInfo(D05roomAll,None,'IBP1')
#
#
# svt03,svd03 = findMachineInfo(D05room0731,'EV1000','SV')
# svt04,svd04 = findMachineInfo(D05room0804,'EV1000','SV')
# svt05,svd05 = findMachineInfo(D05room0805,'EV1000','SV')
# svt06,svd06 = findMachineInfo(D05room0806,'EV1000','SV')
# svt07,svd07 = findMachineInfo(D05room0812,'EV1000','SV')
#
# ibt03,ibd03 = findMachineInfo(D05room0731,None,'IBP1')
# ibt04,ibd04 = findMachineInfo(D05room0804,None,'IBP1')
# ibt05,ibd05 = findMachineInfo(D05room0805,None,'IBP1')
# ibt06,ibd06 = findMachineInfo(D05room0806,None,'IBP1')
# ibt07,ibd07 = findMachineInfo(D05room0812,None,'IBP1')
#
#
#
# a=[i for i in range(len(svt03))]
# b=[i for i in range(len(svt04))]
# c=[i for i in range(len(svt05))]
# d=[i for i in range(len(svt06))]
# e=[i for i in range(len(svt07))]
# for i,j,k,l,m in zip(a,b,c,d,e) :
#     svt03[i] = svt03[i] + datetime.timedelta(hours=9)
#     svt04[j] = svt04[j] + datetime.timedelta(hours=9)
#     svt05[k] = svt05[k] + datetime.timedelta(hours=9)
#     svt06[l] = svt06[l] + datetime.timedelta(hours=9)
#     svt07[m] = svt07[m] + datetime.timedelta(hours=9)
#
# ibt031 = timeChange(ibt03,'UTC')
# ibt041 = timeChange(ibt04,'UTC')
# ibt051 = timeChange(ibt05,'UTC')
# ibt061 = timeChange(ibt06,'UTC')
# ibt071 = timeChange(ibt07,'UTC')
#
# ibptt = timeChange(ibpt,'UTC')
#
# svt0731,svd0731,ibpt0731,ibpd0731=waveMatch(svt03,svd03,ibt031,ibd03)
# svt0804,svd0804,ibpt0804,ibpd0804=waveMatch(svt04,svd04,ibt041,ibd04)
# svt0805,svd0805,ibpt0805,ibpd0805=waveMatch(svt05,svd05,ibt051,ibd05)
# svt0806,svd0806,ibpt0806,ibpd0806=waveMatch(svt06,svd06,ibt061,ibd06)
# svt0812,svd0812,ibpt0812,ibpd0812=waveMatch(svt07,svd07,ibt071,ibd07)
#
# svtime,svdata,ibptime,ibpdata=waveMatch(svt,svd,ibptt,ibpd)
#
# IBP0731={"0731IBPTIME":ibpt0731,"0731IBPDATA":ibpd0731}
# IBP0804={"0804IBPTIME":ibpt0804,"0804IBPDATA":ibpd0804}
# IBP0805={"0805IBPTIME":ibpt0805,"0805IBPDATA":ibpd0805}
# IBP0806={"0806IBPTIME":ibpt0806,"0806IBPDATA":ibpd0806}
# IBP0812={"0812IBPTIME":ibpt0812,"0812IBPDATA":ibpd0812}
#
# saveDic(IBP0731,"IBP0731")
#
# IBPDic={"TIME":ibptime,"DATA":ibpdata}
#

IBPTime,IBPData=findMachineInfo(D05room08,None,'IBP1')
SVTime,SVDATA=findMachineInfo(D05room08,'EV1000','SV')

IBPTimeC = timeChange(IBPTime,'UTC')

IBPStart=timeBinarySearch(IBPTimeC,datetime.datetime.timestamp(IBPTimeC[0]+datetime.timedelta(minutes=30)))
IBPEnd=timeBinarySearch(IBPTimeC,datetime.datetime.timestamp(IBPTimeC[]))



# # 데이터 전처리 후 아래부터는 DB 입력을 위한 작업
#
# testdb,cursor = dbIn()
# IBP0731["0731IBPTIME"][0][0].year
# IBP0731["0731IBPTIME"][0][0].month
# IBP0731["0731IBPTIME"][0][0].day
#
#
#
#
# #id,room_name,date,time,file_name,SV_value
# sql="insert into SVtest_jy(room_name,date,time,file_name,SV_value) values(%s, %s, %s, %s,%s)"
# dd,tt=splitTime(svt0731)
# for i in range(len(svt0731)) :
#     tempStr="/home/wlsdud1512/testNpz001/0731IBP" + str(i+1) +".npz"
#     np.savez(tempStr,Time=IBP0731["0731IBPTIME"][i],Data=IBP0731["0731IBPDATA"]
