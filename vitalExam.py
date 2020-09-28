import time
import os
import numpy as np
from multiprocessing import Pool
import itertools
import re
import vitaldb as vital
import vr_reader_fix as vr

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
    etc=['/mnt/Data/CloudStation/D-05/170714_080052.vital','/mnt/CloudStation/D-05/170714_080052.vital','/mnt/CloudStation/D-01/check_vital_pleth.py','/mnt/CloudStation/D-01/vital_reader.py']

    for i in etc :
        if i in roomInfo :
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
        if tempSplit[4].find(teststr) != -1:
            vrfile.append(i)

    return vrfile

def searchWaveform(data):
    waveForm = ['ECG_II', 'ART1', 'ART2', 'VOLT', 'IBP1', 'IBP3', 'PLETH', 'ECG1', 'CO2', 'AWP','AUDIO','ABP']
    if any(data in word for word in waveForm):
        return True
    else:
        return False

def findMachineInfo(target, machineName=None, machineName2=None):
    count = 0
    # selfData=['/mnt/CloudStation/F-08/200810/F-08_200810_080534.vital', '/mnt/CloudStation/F-08/200810/F-08_200810_104015.vital']
    resultTime = []
    resultData = []

    with Pool(4) as p:
        parResult = p.map(vr.VitalFile, target)

    tempDic={'machineName' : machineName ,'machineName2':machineName2,'vrFile':parResult }
    print(parResult)
    # print(len(parResult))
    for i in parResult:
        time, data = getData(i, machineName, machineName2)
        # print(data[0].shape)
        if time is not None or time != None :
            resultTime=np.append(resultTime,time)
            resultData=np.append(resultData,data)
            count+=1

    return resultTime, resultData

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

# def strBinarySeach(target,data) :

start = time.time()


# roomData=searchRoomAllFile("D-01")

# print(len(roomData))
# print(roomData)
D01room0731=searchDateRoom("D-05",20,8)
E07room0925=searchDateRoom("E-07",20,9,25)

len(D01room0731)
D01room0731

a=vital.VitalFile(D01room0731[0])

vital.load_trk(D01room0731[0])
E07room0925[4]
a=vr.VitalFile(E07room0925[4])

# examFile=['/mnt/CloudStation/D-05/200731/D-05_200731_075952.vital', '/mnt/CloudStation/D-05/200731/D-05_200731_163300.vital']
# a = vital.VitalFile(roomData[2058])

ibpt,ibpd=findMachineInfo(D01room0731,None,'IBP1')

len(ibpt)

with Pool(4) as p :
    a=p.map(vr.VitalFile,D01room0731)

with Pool(4) as p :
    aa=p.map(vital.vital_trks,D01room0731)

aaa=vital.vital_recs(D01room0731[0],aa[0][0])

with Pool(4) as p :
    b=p.map(vital.vital_trks,D01room0731)




# with Pool(4) as p :
#     c=p.map(vital.vital_recs,(D01room0731,d))



print(time.time() - start)