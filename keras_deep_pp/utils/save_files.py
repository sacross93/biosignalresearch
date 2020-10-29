import numpy as np
import os
import utils.datarw as rw
import pandas as pd



def convert_npy1img(filename,x_data):
    import pywt
    #tmp gaus dataimg
    tmp = []
    result = []

    for i in range(len(x_data)//20000):
        for j in range(1,101):
            pcg = x_data[i*20000:(i+1)*20000]
            coef, freqs = pywt.cwt(pcg, j*2, 'cgau1')
            #plt.plot(coef[0])
            #plt.show()

            #make wtimg
            if len(tmp) !=0:
                tmp = np.dstack([tmp, coef[0]])
            elif len(tmp) ==0:
                tmp = coef[0]

        #make serial img
        tmp = np.reshape(tmp,[20000,100,1])
        np.save('/home/jmkim/data/'+filename+"_"+str(i),tmp)

        tmp = []

    return result

def matching_filedata(datapath,file,flag=4):
    meta_datapath = '/home/projects/pcg_transform/pcg_AI/deep/PP/feature_data/'
    meta_files = os.listdir(meta_datapath)
    meta_files.sort()

    wd = pd.read_excel("/home/projects/pcg_transform/PYS/phase.xlsx")
    phase_data = np.array(wd)

    liver_out = -1
    reperfusion = -1
    for i in range(len(phase_data)):
        if str(phase_data[i, 0]) == file[:6]:
            liver_out = phase_data[i, 1]
            reperfusion = phase_data[i, 2]


    x_data = rw.read_signal_data(datapath + file)
    x_data = x_data.reshape(len(x_data) // (20 * 1000), 20 * 1000)
    # meta file read

    m_file = ''
    for fcnt in meta_files:
        if fcnt.find('result_data.csv') > 0 and fcnt.find(file[:6]) >= 0:
            print(fcnt)
            m_file = fcnt[:]


    ws = pd.read_csv(meta_datapath + m_file,engine='python')
    ws = ws[['time','PP']]
    ws = ws.dropna()
    txy = np.array(ws)

    st = 0
    tmp = -1
    for i in range(len(txy)):
        if txy[i, 0] < str(liver_out):
                continue

        elif txy[i, 0] < str(reperfusion):
            if flag == 0:
                tmp = i
                break
        else:
            if flag == 1:
                tmp = i
                break
            elif flag == 2:
                st = i
                break


    y_data = txy[st:tmp,1]
    y_data = y_data.astype('float32')
    x_data = x_data[st:tmp]

    return x_data.flatten(), y_data

def save_data(datapath,file,flag):


    x_data,y_data = matching_filedata(datapath ,file, flag )
    #x_data = rw.read_signal_data(datapath + file)

    filename = file[:6] + "_" + str(flag)
    convert_npy1img(filename,x_data)
    #np.save(file+"_wtimg_"+str(flag),data)
    np.save('/home/jmkim/ydata/'+filename+"_y_data",y_data)

"""
datapath = '/home/projects/pcg_transform/pcg_AI/deep/PP/pcg_data/'
files = os.listdir(datapath)
files.sort()


set_file= []
for fcnt in files:
    if fcnt.find('svr.') <0:
        #if fcnt.find('180608') <0 and fcnt.find('180614') <0 and fcnt.find('180626') <0  and fcnt.find('180601') <0:
        set_file.append(fcnt)



for file in set_file:


    print(file)


    for flag in range(3):
        save_data(datapath,file,flag)
    #np.load(outputpath+file+"_wtimg.npy")




#data.shape




"""