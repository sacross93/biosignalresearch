import pymysql
from DB_save.DB_util import *
import matplotlib.pyplot as plt
import os
# def hypotension_pid_intellivue():

import pyvital2 as pyvital
import csv
import vr_reader_fix as vr
import os
from scipy import signal
import matplotlib.pyplot as plt
from anssignal import *
import pymysql
import check_filepath
from sklearn.preprocessing import minmax_scale

def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
# d6 end -> d2 start

def get_parameters(filename):


    vrfile = vr.VitalFile(filename)
    IBPflag = 0
    ABPflag = 0
    IBPname = ''
    ecg_name = ''
    for trk in vrfile.trks.values():  # find event track
        # print(trk['name'],tname)
        if trk['name'].find('IBP1')>=0:
            IBPflag = 1
            IBPname = 'IBP1'
            break
        elif trk['name'].find('IBP5')>=0:
            IBPflag = 1
            IBPname = 'IBP5'
            break

    for trk in vrfile.trks.values():  # find event track
        # print(trk['name'],tname)
        if trk['name'].find('ART1_SBP')>=0:
            ABPflag = 1

    if IBPflag == 0 or ABPflag == 0:
        print('ABP not Exist')
        return [],[],[]
    result = []

    result.append(vrfile.get_samples(IBPname))
    result.append(vrfile.get_samples('PLETH'))
    result.append(vrfile.get_samples('ECG1'))
    result.append(vrfile.get_numbers('HR'))
    result.append(vrfile.get_numbers('ART1_MBP'))
    result.append(vrfile.get_numbers('ART1_DBP'))
    result.append(vrfile.get_numbers('ART1_SBP'))
    time_dbp, dbp = vrfile.get_numbers('ART1_DBP')
    time_sbp, sbp = vrfile.get_numbers('ART1_SBP')
    if len(dbp) > len(sbp):
        map = (np.array(dbp[:len(sbp)])+np.array(sbp))/2
        time = time_sbp
    else:
        map = (np.array(dbp) + np.array(sbp[:len(dbp)])) / 2
        time = time_dbp



    return np.array(result),np.array(time),map

room_names = ['C-05','C-06','B-01','B-02','B-03','B-04']
room_names = ['C-04','C-01']
# startdays = range(190201,190727)
# startday = '200217'
# '190430','190403','190416','190425','190502',

startdays = []
for i in range(700):
    tmp = 200100+i
    startdays.append(str(tmp))


day = []

for room_name in room_names:
    for startday in startdays:
        startday = str(startday)

        # if day[i].find('180510')==-1:
        #    continue
        filenames = check_filepath.search_filepath([room_name], [startday])
        filenames = list(filenames)
        filenames.sort()

        #    continue
        filenames2 = []
        cnt = 0

        cnt = 0
        for j in range(len(filenames)):
            if (os.path.getsize(filenames[j]) > 5000000):
                filenames2.append(filenames[j])
                print(filenames[j])
            else:
                print("small file")

        if len(filenames2) == 0:
            continue

        filename = filenames2

        # file = filename2[1]

        for file in filename:


            vrfile = vr.VitalFile(file)

            result,map_time, total_map = get_parameters(file)

            # fig = plt.figure(figsize=(20, 10))
            # plt.plot(map[:, 0], map[:, 1])
            # plt.title('MAP')
            # plt.savefig('/home/projects/SVV/hypotension/' + str(p) + '.png')
            # plt.close()

            if len(total_map) >1200 and np.mean(total_map)>65:
                map = total_map[600:-600]
                flag65 = 0
                i = 0
                while( i<len(map)):
                    if map[i] < 65:
                        flag65 += 1
                        i += 200
                    i = i+1
                    # print(map)

                if flag65>=5:

                    FILE_DATE = file[-19:-13]
                    DATE_TIME = file[-12:-6]
                    room_name = file[-24:-20]

                    file_name = file.split('/')[-1].split('.')[0]

                    os.system("cp "+ file+" /home/projects/SVV/hypotension_vital/" )

                    # result.append(file)
                    fig = plt.figure(figsize=(20, 10))
                    plt.plot(map_time[600:-600],map)
                    plt.title(file.split('/')[-1])
                    plt.savefig('/home/projects/SVV/hypotension/' + file_name + '.png')
                    plt.close()
                    print(file.split('/')[-1])

                    np.savez("/home/projects/SVV/hypotension_file/" +file_name +".npz")

        # full_filename = file_path + File_date + "/" + filename

    # import os
    # os.mkdir('/home/projects/SVV/hypotension_vital/')