import vr_reader_fix as vr
import multiprocessing
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
    file = []
    result= []
    while True:
        if roomInfo is None or not roomInfo:
            break
        else:
            temp=roomInfo.pop()
            if temp == '/mnt/Data/CloudStation/D-05/170714_080052.vital' :
                temp=roomInfo.pop()
            temp = search(temp)
            for i in range(len(temp)):
                file.append(temp.pop())

    while True :
        if file is None or not file :
            break
        else :
            temp=file.pop()
            if os.path.getsize(temp) > 5000000 :
                result.append(temp)
    result.sort()
    return result

def searchDate(roomName,year, month=None, day=None):
    vrfile=[]
    for i in roomName :
        if year < int(i[28:30]) :
            break
        if year == int(i[28:30]) :
            if month != None and day != None :
                if month == int(i[30:32]) and day == int(i[32:34]) :
                    vrfile.append(i)
            elif month != None and day == None :
                if month == int(i[30:32]) :
                    vrfile.append(i)
            elif month == None and day != None :
                if str(day) in int(i[32:34]) :
                    vrfile.append(i)
            else :
                vrfile.append(i)
    return vrfile

def searchWaveform(data) :
    waveForm = ['ECG_||', 'ART1', 'ART2', 'VOLT', 'IBP1', 'IBP3', 'PLETH', 'ECG1', 'CO2', 'AWP']
    if any(data in word for word in waveForm) :
        return True
    else :
        return False

def findMachineInfo(target,machineName=None,machineName2=None):
    time=[]
    data=[]
    count=0
    resultTime=[]
    resultData=[]

    num_cores = multiprocessing.cpu_count()
    parResult = parmap.map(vr.VitalFile,target,pm_pbar=True,pm_processes=num_cores)
    print(parResult)
    # print(len(parResult))

    for i in parResult :
        time,data=getData(i,machineName,machineName2)
        resultTime.append(np.append(resultTime,time))
        resultData.append(np.append(resultData,data))

    return resultTime,resultData

def getData(vrfile,machineName=None,machineName2=None) :
    # count=0
    # print(vrfile)
    # print(machineName)
    # print(machineName2)
    Dtime=[]
    Ddata=[]
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
                        else:
                            # print("correct")
                            Dtime, Ddata = vrfile.get_numbers(machineName2, machineName)

        else:
            if machineName2 is not None:
                if trk['name'].find(machineName2):
                    if searchWaveform(machineName2):
                        # print(len(trk['name']))
                        Dtime, Ddata = vrfile.get_samples(machineName2)
                    else:
                        Dtime, Ddata = vrfile.get_numbers(machineName2)
    # print(Ddata)

    return Dtime, Ddata

def isCorrect(fileName):  # ABP waveform , EV1000 SV search function
    vrFile = vr.VitalFile(fileName)
    time, date = vrFile.get_numbers('SV', 'EV1000')
    for trk in vrFile.trks.values():
        if trk['name'].find('IBP1') >= 0:
            result = 1
            break
        elif trk['name'].find('IBP5') >= 0:
            result = 2
            break
        elif len(time) != 0 and len(date) != 0:
            result = 3
            break

    return result

def statistics(data,require=None) :
    average=np.mean(data)
    std=np.std(data)
    var=np.var(data)
    # print("average= ",average," std= ",std," var= ",var)
    if require is not None :
        if require == 'average' :
            return average
        if require == 'std' :
            return std
        if require == 'var' :
            return var
    else :
        return average, std, var

def quartile(temp):  # quartile function
    data=np.array([])
    data=np.append(data,temp)
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
    Q=quartile(data)
    IQR = Q[2] - Q[0]
    result = [Q[0] - (1.5 * IQR), Q[2] + (1.5 * IQR)]
    return result

def plot(time,data) :
    print("plot start")
    newData = np.array([])
    newTime = np.array([])
    newData = np.append(newData, data)
    newTime = np.append(newTime, time)
    print("outLier in")
    Q=outLier(newData)
    print("plot in")
    plt.plot(newTime, newData)
    plt.axhline(y=statistics(newData, 'average'), color='r', linewidth=1)

    # num_cores = multiprocessing.cpu_count()
    # temp=np.array_split(data,num_cores)
    # parResult = parmap.map(a, temp, pm_pbar=True, pm_processes=num_cores)

    for i in range(len(data)) :
        print("for in")
        if Q[0] > data[i] or Q[1] < data[i] :
            print("if in")
            plt.axvline(x=newTime[i], color='r', linestyle='--', linewidth=0.5)

    # print(Q[0],Q[1])
    plt.show()
    plt.close()

def timeBinarySearch(timeLength, value):
    low = 0
    high = len(timeLength) - 1
    value = int(datetime.datetime.timestamp(value))
    while (low <= high):
        mid = int((low + high) / 2)
        if int(datetime.datetime.timestamp(timeLength[mid])) > value:
            high = mid - 1
        elif int(datetime.datetime.timestamp(timeLength[mid])) < value:
            low = mid + 1
        else:
            return timeLength[mid],mid
    return None,None

def timeChange(time,timeType,lateTime=None) :
    tempTime=[]
    if timeType == 'timestamp' :
        for i in range(len(time)) :
            if lateTime is None :
                tempTime.append(datetime.datetime.timestamp(time[i]))
            else :
                tempTime.append(datetime.datetime.timestamp(time[i] + datetime.timedelta(hours=lateTime)))
    elif timeType == 'UTC' :
        for i in range(len(time)) :
            if lateTime is None :
                tempTime.append(datetime.datetime.fromtimestamp(time[i]))
            else :
                tempTime.append(datetime.datetime.timestamp(time[i] + datetime.timedelta(hours=lateTime)))

    return tempTime

def timeMatchData(time1,data1,time2,data2,btime=None) :
    resultTime1=np.array([])
    resultTime2=np.array([])
    matchData1=np.array([])
    matchData2=np.array([])
    matchtime = time1[0].second % 10
    for i in range(len(time1)) :
        # print("for")
        if time1[i].second % 10 == matchtime :
            # print("if")
            tempValues,j=timeBinarySearch(time2,time1[i])
            # print("if2")
            if tempValues is not None :
                # print(j)
                # print(len(data2))
                # print("1")
                matchData1=np.append(matchData1,data1[i])
                # print("2")
                resultTime1=np.append(resultTime1,time1[i])
                # print("3")
                matchData2=np.append(matchData2,data2[j])
                # print("4")
                resultTime2=np.append(resultTime2,time2[j])
                # print("5")

    # print("out")
    return resultTime1,matchData1,resultTime2,matchData2

def saveData(data,name) :

    np.savez("/home/wlsdud1512/testNpz001/" + name + ".npz", data)




roomName = searchRoomAllFile("F-08")
#2020-08-25
F08room2020 = searchDate(roomName, 20, 8, 25)
print(F08room2020)
atime, adata = findMachineInfo(F08room2020, None, 'AUDIO')

print("finish")