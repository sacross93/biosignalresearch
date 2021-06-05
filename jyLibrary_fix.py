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

cpu = multiprocessing.cpu_count()

#start read vital file
def search(dirname,address=None):
    full_filename = []
    if address == None :
        address = ("/mnt/Data/CloudStation/")
    if dirname.find("/mnt/Data/CloudStation/") == -1 :
        dirname = address + dirname

    filenames = os.listdir(dirname)

    # with Pool(cpu//2) as p :
    #     full_filename=p.map(os.path.join,(dirname,filenames))

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
        teststr += str(month.zfill(2))
    if day != None :
        teststr += str(day.zfill(2))

    # teststr = teststr.replace("None", "")

    # for i in roomDir :
    #     tempSplit=i.split('/')
    #     if tempSplit[5].find(teststr) != -1:
    #         vrfile.append(i)

    with Pool(cpu//2) as p :
        vrfile = p.map(split_day,roomDir)

    return vrfile

def split_day(teststr) :
    tempSplit=teststr.split('/')
    if tempSplit[5].find(teststr) != -1 :
        return teststr
    else :
        return None
