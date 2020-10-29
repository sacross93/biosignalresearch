from utils.my_classes import DataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from utils.processing import *
from utils.models import *
import pymysql
import os
from tcn.tcn import  compiled_tcn

type = 'result_3ch_dataset_result'

folder_name = 'result_3ch_dataset_result/'
chnnal = 3

savepath = '/home/projects/pcg_transform/pcg_AI/deep/SVV/result/'+folder_name

if not os.path.isdir(savepath ):
    os.mkdir(savepath )



stdate = 180201
enddate = 180232



conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                       db='abp_svv_generator2', charset='utf8')
curs = conn.cursor()

sql = """select file_name,EV_SVV from abp_svv_tableset where date >=%s and date <=%s and type = %s order by date,file_name;"""
curs.execute(sql,(stdate,enddate,type))
row = curs.fetchall()

training_data = np.array(row)[:,0]
labels = {row[i][0]: row[i][1] for i in range(len(row))}

conn.close()


data = np.load(training_data[3000])

fig = plt.figure(figsize=(20, 5))
plt.subplot(1,3,1)
plt.plot(data[:,0][:500],color = 'red')
plt.title('ABP waveform', size = 20)
plt.xticks([])
plt.yticks([])

plt.subplot(1,3,2)
plt.plot(data[:,1][:500],color = 'red')
plt.title('Slope of ABP waveform', size = 20)
plt.xticks([])
plt.yticks([])

plt.subplot(1,3,3)

plt.plot(data[:,2][250:750],color = 'red')
plt.title('Frequency of ABP', size = 20)
plt.xticks([])
plt.yticks([])

savepath = '/home/projects/SVV/'
plt.savefig(savepath + 'fig1.png',dpi=300)
plt.close()

fig = plt.figure(figsize=(20, 5))
plt.plot(data[:,1],color = 'red')
plt.title('Slope of ABP waveform', size = 20)
plt.xticks([])
plt.yticks([])
savepath = '/home/projects/SVV/'
plt.savefig(savepath + 'fig3.png',dpi=300)
plt.close()

