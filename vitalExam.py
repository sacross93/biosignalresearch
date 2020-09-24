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

    for i in roomDir :
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
    count = 0
    # selfData=['/mnt/CloudStation/F-08/200810/F-08_200810_080534.vital', '/mnt/CloudStation/F-08/200810/F-08_200810_104015.vital']
    resultTime = []
    resultData = []

    with Pool(4) as p:
        parResult = p.map(vital.VitalFile, target)

    tempDic={'machineName' : machineName,'machineName2':machineName2,'vrFile':parResult }
    print(parResult)
    # print(len(parResult))
    for i in parResult:
        time, data = getData(i, machineName, machineName2)
        # print(data[0].shape)
        if time is not None or time != None :
            resultTime=np.append(resultTime,time,axis=count)
            resultData=np.append(resultData,data,axis=count)
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

# examFile=['/mnt/CloudStation/D-05/200731/D-05_200731_075952.vital', '/mnt/CloudStation/D-05/200731/D-05_200731_163300.vital']
# a = vital.VitalFile(roomData[2058])

ibpt,ibpd=findMachineInfo(D01room0731,None,'IBP1')

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