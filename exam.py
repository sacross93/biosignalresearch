import vr_reader_fix as vr
import multiprocessing
import pymysql
from multiprocessing import *
import itertools
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
from scipy.signal import butter, lfilter, hilbert, chirp ,find_peaks
from scipy import signal
from scipy.interpolate import interp1d


#start read vital file
def search(dirname):
    full_filename = []
    addressTemp = ("/mnt/Data/CloudStation/")
    if dirname.find("/mnt/Data/CloudStation/") == -1 :
        dirname = addressTemp + dirname

    filenames = os.listdir(dirname)

    for filename in filenames:
        full_filename.append(os.path.join(dirname, filename))
    return full_filename

def searchRoomAllFile(roomname):
    roomInfo = search(roomname)
    # print(roomInfo)
    etc=['/mnt/Data/CloudStation/D-05/170714_080052.vital','/mnt/Data/CloudStation/D-05/170714_080052.vital','/mnt/Data/CloudStation/D-01/check_vital_pleth.py','/mnt/Data/CloudStation/D-01/vital_reader.py']
    for i in etc :
        if i in roomInfo :
            # print("delete : ",i)
            roomInfo.remove(i)

    with Pool(4) as p :
        file = p.map(search,roomInfo)

    file = list(itertools.chain(*file))

    file = list(file)
    for i in file :
        if os.path.getsize(i) <= 5000000 :
            file.remove(i)

    file.sort()
    return file

def searchDateRoom(roomname,year,month=None,day=None) :
    roomDir=searchRoomAllFile(roomname)
    vrfile=[]

    teststr = str(year)

    if month != None :
        if month // 10 == 0 :
            teststr += str(0)+str(month)
        else :
            teststr += str(month)
    if day != None :
        if day // 10 == 0 :
            teststr += str(0)+str(day)
        else :
            teststr += str(day)

    teststr = teststr.replace("None", "")

    for i in roomDir :
        tempSplit=i.split('/')
        if tempSplit[5].find(teststr) != -1:
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

    dtname=vrfile.vital_trks()
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
#end read vital file


#start Data analysis
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
    plt.subplot(1,2,1)
    plt.plot(newTime, newData)
    plt.axhline(y=statistics(newData, 'average'), color='r', linewidth=1)
    peaks, _ = find_peaks(data,distance=4000,threshold=0.8)
    plt.subplot(1,2,2)
    plt.plot(peaks,data[peaks],"xr")
    plt.plot(data)
    plt.legend(['prominence'])
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

def butter_bandpass(lowcut, highcut, fs, order=5) :
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b,a

def butter_bandpass_filter(data,lowcut,highcut,fs,order=5) :
    print("filter start")
    b, a = butter_bandpass(lowcut,highcut,fs,order=order)
    print("bandpass...")
    return lfilter(b,a,data)

def notch_pass_filter(data,center,interval=20,sr=44100,normalized=False) :
    center = center/(sr/2) if normalized else center
    b,a = signal.irrnotch(center,center/interval,sr)
    filtered_data = signal.lfilter(b,a,data)
    return filtered_data
#end Data analysis

#start DB process
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
#end DB process



start = time.time()

D01room0826=searchDateRoom("D-01",20,8,26)
print("loadiong...")

adt,add=findMachineInfo(D01room0826,None,"AUDIO")

adt = timeChange(adt,"UTC")

tempTime=datetime.datetime(2020,8,26,9,33,40)
tempTime=datetime.datetime.timestamp(tempTime)

tempTime2=datetime.datetime(2020,8,26,9,33,50)
tempTime2=datetime.datetime.timestamp(tempTime2)

startTime=timeBinarySearch(adt,tempTime)
endTime=timeBinarySearch(adt,tempTime2)

tempadt=adt[startTime:endTime]
tempadd=add[startTime:endTime]


#interpid scipy


#--------------------------------------
#band pass fillter

b=butter_bandpass_filter(tempadd,300,1900,4000)
bb=butter_bandpass_filter(tempadd,10,100,4000)

# plt.figure(figsize=(20, 10))
# plt.plot(tempadd[25000:30000])
# plt.show()
#
# plt.figure(figsize=(20, 10))
# plt.plot(tempadd[25000:30000])
# plt.plot(b[25000:30000])
# plt.plot(bb[25000:30000])
# plt.show()
#
# plt.close()

t=np.arange(1*4000)/4000
analytic_signal = hilbert(bb)
amplitude_envelope = np.abs(analytic_signal)

instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency=(np.diff(instantaneous_phase) / (2.0*np.pi) * 4000 )

fig=plt.figure(figsize=(20,10))
ax0=fig.add_subplot(211)
ax0.plot(t,bb[2000:6000],label='signal')
ax0.plot(t,amplitude_envelope[2000:6000],label='envelope')
ax0.legend()

ax1=fig.add_subplot(212)
ax1.plot(t,instantaneous_frequency[2000:6000])
ax1.set_xlabel("time in seconds")
ax1.set_ylim(0.0,150.0)
plt.show()

plt.figure(figsize=(20,10))
peaks, _ = find_peaks(bb, distance=1000,threshold=0.8,prominence=1)
plt.plot(bb)
plt.plot(peaks,bb[peaks],"x")
plt.show()
plt.close()

fig=plt.figure(figsize=(20,10))
ax0=fig.add_subplot(211)
ax0.plot(t,bb[2000:6000],label='signal')
ax0.plot(t,amplitude_envelope[2000:6000],label='envelgope')
ax0.legend()
ax1=fig.add_subplot(212)
ax1.plot(bb)
ax1.plot(peaks,bb[peaks],"x",label="peak")
plt.show()



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

print(time.time() - start)