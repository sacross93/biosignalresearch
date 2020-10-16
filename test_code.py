import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import itertools
from multiprocessing import Pool
import vr_reader_fix as vr
from scipy.signal import hilbert, chirp
import vitaldb as vital

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
        if datetime.datetime.timestamp(time[mid]) > target :
            # print("2")
            high = mid - 1
        elif datetime.datetime.timestamp(time[mid]) < target :
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
    etc=['/mnt/Data/CloudStation/D-05/170714_080052.vital','/mnt/Data/CloudStation/D-05/170714_080052.vital','/mnt/Data/CloudStation/D-01/check_vital_pleth.py','/mnt/Data/CloudStation/D-01/vital_reader.py','/mnt/Data/CloudStation/D-06/170608_074619.vital']
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

print("start")
tempVrfile=searchDateRoom("D-01",20,8,26)

print("loading")
tempVital=vr.VitalFile(tempVrfile[0])
dtname = tempVital.vital_trks()

a="AUDIO"
bb,bbb=tempVital.get_samples("AUDIO")


tempVrfile=vital.VitalFile(tempVrfile[0],sels=dtname[0])

dt,dd=tempVrfile.get_samples(dtname[0],interval=0.0008)
len(dd)


duration = 1.0
fs = 400.0
samples = int(fs*duration)
t = np.arange(samples) / fs
signal = chirp(t, 20.0, t[-1], 100.0)
signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )


analytic_signal = hilbert(signal)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = (np.diff(instantaneous_phase) /
                            (2.0*np.pi) * fs)
fig = plt.figure()
ax0 = fig.add_subplot(211)
ax0.plot(t, signal, label='signal')
ax0.plot(t, amplitude_envelope, label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()
ax1 = fig.add_subplot(212)
ax1.plot(t[1:], instantaneous_frequency)
ax1.set_xlabel("time in seconds")
ax1.set_ylim(0.0, 120.0)
plt.show()