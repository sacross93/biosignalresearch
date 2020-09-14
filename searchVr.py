import vr_reader_fix as vr
import multiprocessing
import parmap
from functools import partial
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

def search(dirname) :
    full_filename=[]
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename.append(os.path.join(dirname, filename))
    return full_filename

def searchRoom(roomname) : # room number folder search function
    full_filename=[]
    filenames = os.listdir(roomname)
    for filename in filenames :
        if '-' in filename :
            if filename.find("-") == 1 :
                full_filename.append(os.path.join(roomname,filename))
    return full_filename

def isCorrect(fileName) : # ABP waveform , EV1000 SV search function
    result=False
    vrFile=vr.VitalFile(fileName)
    time, date = vrFile.get_numbers('SV','EV1000')
    for trk in vrFile.trks.values() :
        if trk['name'].find('IBP1')>=0 :
            result = True
            break
        elif trk['name'].find('IBP5')>=0 :
            result = True
            break
        elif len(time)!=0 and len(date)!=0 :
            result = True
            break
    return result

def quartile(data) : # quartile function
    Q1 = len(data) * 0.25
    if Q1 % 1 != 0 :
        Q1 = (data[int(Q1)] + data[int(Q1) - 1]) / 2
    else :
        Q1 = data[int(Q1)-1]

    Q2 = len(data) * 0.5
    if Q2 % 1 == 0 :
        Q2 = (data[int(Q2)] + data[int(Q1)-1]) / 2
    else :
        Q2 = data[int(Q2)+1]

    Q3 = len(data) * 0.75
    if Q3 % 1 != 0 :
        Q3 = (data[int(Q3)] + data[int(Q3) - 1]) / 2
    else :
        Q3 = data[int(Q3)-1]

    Q4 = data[int(len(data)*1)-1]

    return Q1,Q2,Q3,Q4

def outLier(Q) : # box plot outLier search
    IQR = Q[2] - Q[0]
    result = [Q[0] - (1.5 * IQR), Q[2] + (1.5 * IQR)]
    return result

def timeBinarySearch2(timeLength,value) :
    low = 0
    high = len(timeLength)-1
    value=int(datetime.datetime.timestamp(value))
    while(low <= high) :
        mid = int((low + high ) / 2)
        if int(datetime.datetime.timestamp(timeLength[mid])) > value :
            high = mid - 1
        elif int(datetime.datetime.timestamp(timeLength[mid])) < value :
            low = mid + 1
        else :
            return timeLength[mid]
    return None


def timeBinarySearch(timeLength,value) : # IBP1 , SV 1second match function

    binaryTemp=int(len(timeLength)/2)
    result = False
    left=0
    right=0
    count=0


    while(left>0 or right<=len(timeLength) or left < right or binaryTemp != 0) :
        if value.hour < timeLength[binaryTemp].hour :
            right=binaryTemp-1
            binaryTemp -= int(binaryTemp/2)
        elif value.hour > timeLength[binaryTemp].hour :
            left=binaryTemp+1
            binaryTemp += int(binaryTemp/2)
        elif value.hour == timeLength[binaryTemp].hour :
            if value.minute < timeLength[binaryTemp].minute :
                right=binaryTemp-1
                binaryTemp -= int(binaryTemp/2)
            elif value.minute > timeLength[binaryTemp].minute :
                left=binaryTemp+1
                binaryTemp += int(binaryTemp/2)
            elif value.minute == timeLength[binaryTemp].minute :
                if value.second < timeLength[binaryTemp].second :
                    right = binaryTemp-1
                    binaryTemp -= int(binaryTemp / 2)
                elif value.second > timeLength[binaryTemp].second :
                    left = binaryTemp+1
                    binaryTemp += int(binaryTemp / 2)
                elif value.second == timeLength[binaryTemp].second :
                    print(timeLength[binaryTemp])
                    result == True
                    return timeLength[binaryTemp]
                    break
        else :
            continue

    if result == True :
        return timeLength[binaryTemp]
    else :
        return None

# aa=searchRoom("/mnt/Data/CloudStation/")
a=search("/mnt/Data/CloudStation/D-05")
for i in range(len(a)) :
    if a[i] == '/mnt/Data/CloudStation/D-05/170714_080052.vital' :
        print(i)
        a.pop(i)
        break

b=[]
while(True) :
    if a is None or not a :
        break
    else :
        c=search(a.pop())
        for i in range(len(c)) :
            b.append(c.pop())
print(a)
b

accurateFile=[]
accurateFile2=[]
b.sort()
for i in range(len(b)) :
    if os.path.getsize(b[i]) > 5000000 :
        accurateFile.append(b[i])
        # if isCorrect(b[i]) :
        #     accurateFile2.append(b[i])
        #     print(b[i])

print("raw")

print(len(b))


print("")
print("refine")
print(len(accurateFile))
print("")

accurateFile.sort()
date=[]
roomName=[]
accurateFile

for i in range(30) :
    temp=accurateFile.pop()
    if '2008' in temp :
    # if isCorrect(temp) : # -> True
        date.append(temp[28:34])
        roomName.append(temp[23:27])

print(date,roomName)

vrfile=cf.search_filepath(roomName,date)
vrfile
vrfile2 = vrfile.pop()
vrfile2
vrfileTemp = vr.VitalFile(vrfile2)
vrfile2
print(vrfileTemp)
vrfile=list(vrfile)
vrfileList=[]
num_cores=multiprocessing.cpu_count() # cpu count
pool = multiprocessing.Pool(num_cores) # using process
splited_data = np.array_split(vrfile,num_cores)
splited_data = [x.tolist() for x in splited_data]
vrfileList=parmap.map(vr.VitalFile,splited_data,pm_pbar=True, pm_processes=num_cores)


while(True) :
    time,temp=vrfileTemp.get_numbers('SV','EV1000')
    if len(time)!=0 and len(temp)!=0 :
        break
    else :
        vrfile2 = vrfile.pop()
        vrfileTemp = vr.VitalFile(vrfile2)
while(True) :
    time2, temp2=vrfileTemp.get_samples('IBP1')
    if len(time2)!=0 and len(temp2)!=0 :
        break
    else :
        vrfile2 = vrfile.pop()
        vrfileTemp = vr.VitalFile(vrfile2)
# while(True) :
#     time3, temp3=vrfileTemp.get_numbers('IBP5') # not exist
#     if len(time3)!=0 and len(temp3)!=0 :
#         break
#     else :
#         vrfile2 = vrfile.pop()
#         vrfileTemp = vr.VitalFile(vrfile2)


# print(time)
# print(temp)


# print(len(temp2))
# print(time2)

# print(temp3[1]) #IBP5 is null..D-2 2007~2008
plt.plot(time,temp)
# plt.plot(time2[50000:50000+1000],temp2[50000:50000+1000])
plt.show()
plt.close()
# len(time2)

IBPtime=[]
for i in range (len(time2)) :
    IBPtime.append(datetime.datetime.fromtimestamp(time2[i]))
SVtime=[]
SVtime2=[]
for i in range(len(time)) :
    SVtime.append(time[i]+datetime.timedelta(hours=9))
    SVtime2.append(datetime.datetime.timestamp(time[i]+datetime.timedelta(hours=9)))
len(IBPtime)
len(SVtime)
"""
hourCnt=0
minuteCnt=0
for i,j in zip(time,IBPtime) : # compare EV1000 to IBP1
    if i.hour == j.hour :
        print("hour pass")
        hourCnt += 1
        if i.minute == j.minute :
            minuteCnt += 1
            if i.second == j.second :
                print(i)
                print(j) # ..... fail

print(hourCnt) # 1790
print(minuteCnt) # 30
#EV1000 SV , IBP1 equal time check
"""

average = 0
for i in range(len(temp)) :
    average += temp[i]
average /= len(temp)
#average check

sortTemp=copy.deepcopy(temp)
sortTemp.sort()
# sortTemp
# temp
Q=np.zeros(20)
Q=quartile(sortTemp)
Q

# IQR = Q[2]-Q[0]
# IQR
# outLier = [ Q[0]-(1.5*IQR) , Q[2]+(1.5*IQR) ]
# outLier

OL=outLier(Q)
OL
std = np.std(temp)
plt.plot(time,temp)
plt.axhline(y=average, color='r', linewidth=1) # average line

for i in range(len(temp)) : # all_outLier search
    if temp[i] > OL[1] or temp[i] < OL[0] :
        plt.axvline(x=time[i], color='r', linestyle='--', linewidth=0.5)

plt.show()
plt.close()
# outLier check

# for i in range(len(temp)) :
 # if temp[i] > OL[1] or temp[i] < OL[0] :
     # time.pop(i)
     # temp.pop(i)
# plt.plot(time,temp)
# plt.axhline(y=average, color='r', linewidth=1)
# plt.show()
# plt.close()

# remove outLier

# test
matchData=np.array([])

for i in range(len(SVtime)) :
    if SVtime[i].second % 10 == 6 :
        tempValue=(timeBinarySearch2(IBPtime,SVtime[i]))
        if tempValue != None :
            matchData=np.append(matchData,tempValue)
            tempValue = None
        else :
            continue

matchData
# np.savez("/home/wlsdud1512/testNpz001/"+"20200814IBPSV"+".npz",matchData)
# testNp=np.load("/home/wlsdud1512/testNpz001/20200814IBPSV.npz",allow_pickle=True)
# testNp["arr_0"]
print("finish")