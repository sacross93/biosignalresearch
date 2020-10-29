import sklearn
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt


import csv

import keras
#from SVV_projects.read_SVV import read_abp_svv, read_abp_svv_10sec,read_abp_svv_minmax_fft
from PCG_SVV.pcg_models import *

#import sklearn.metrics
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D,Dense,LSTM,Dropout
import pymysql
import os
from tcn.tcn import  compiled_tcn
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,3";

type = 'minmax'


HOSTNAME = 'localhost'
USERNAME = 'jmkim'
PASSWORD = 'anesthesia'
DBNAME = 'data_generator'
DEVICE_DB_NAME = 'Vital_DB'


date = 190611

conn = pymysql.connect(host='localhost', user=USERNAME, password=PASSWORD,
                       db=DBNAME, charset='utf8')
curs = conn.cursor()

sql = """select time,pcg_file,SVV from data_generator.PCG_SVV where date =%s and type = %s and room_name = 'D-02' order by date,pcg_file;"""
curs.execute(sql,(date,type))
row = curs.fetchall()
conn.close()



import datetime



training_data = np.array(row)[:,1]
time =  np.array(row)[:,0]
svv =np.array(row)[:,2]

hour = 9
cnt = 0
for i in range(10,len(time)):
    if int(time[i][11:13])==hour:
        cnt = i
        break

cnt = cnt+306
svv[cnt]
dt = time[cnt]
print(dt)
data = np.load(training_data[cnt])
data= data['arr_0']


fig = plt.figure(figsize=(20, 10))

plt.subplot(3,1,1)

plt.plot(data[:,:,0][0])

plt.subplot(3,1,2)
plt.plot(data[:,:,1][0])
plt.xlabel(dt)

#histogram
plt.subplot(3,1,3)
testdata = data[:,:,1][0]
mean = np.mean(testdata)
std = np.std(testdata)

fftdata = np.fft.fft(testdata)

plt.plot(procdata)
plt.show()

result = np.abs(mean - testdata)/std
# a = np.argmax(result)
# result[a] = 0
plt.plot(result[result>4])

plt.show()


#
# plt.show()
# plt.hist(result,bins=100)
# plt.show()