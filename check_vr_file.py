#import pyvital2 as pyvital
#import csv
import vr_reader_fix as vr
import os
from scipy import signal
import matplotlib.pyplot as plt
from anssignal import *
#import pymysql
import check_filepath
#from sklearn.preprocessing import minmax_scale

art_matching = {'D-02': 'ART1', 'D-06': 'ART2'}


def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
# d6 end -> d2 start

def read_D2_PCG(room_name, filename):
    time_PCG = np.array([])
    PCG= np.array([])
    for f in filename:

        vrfile = vr.VitalFile(f)
        svvflag = 0
        pcgflag = 0
        ecg_name = ''
        for trk in vrfile.trks.values():  # find event track
            # print(trk['name'],tname)
            dname = vrfile.devs[trk['did']]
            if dname['name'] == 'EV1000':

                if trk['name'].find('SVV')>=0:
                    svvflag = 1
                    break

        if svvflag == 0:
            print('SVV not Exist')
            return [],[],[],[]

        for trk in vrfile.trks.values():  # find event track
            # print(trk['name'],tname)
            dname = vrfile.devs[trk['did']]
            if dname['name'] == 'DI-155':

                if trk['name'].find('VOLT')>=0:
                    pcgflag = 1
                    break

        if pcgflag == 0:
            return [],[],[],[]

        time_PCG, PCG = vrfile.get_samples('VOLT')
        time_ABP, ABP = vrfile.get_samples('ART2')


    return time_PCG, PCG,time_ABP, ABP


#
# z = 1

# wd = pd.read_excel("/home/projects/pcg_transform/PYS/phase.xlsx")
# phase_data = np.array(wd)

now = str(datetime.datetime.now() - datetime.timedelta(days=1))
yesterday = now[2:4] + now[5:7] + now[8:10]

room_name = 'K-03'
# startdays = range(190201,190727)
startdays = [200730]
# '190430','190403','190416','190425','190502',


# startdays = ['190530','190531','190605','190607','190612','190613','190617','190619','190620']
# startdays =  ['190521','190523','190604','190531','190530','190527']
# startdays = ['190503', '190426', '190425', '190424', '190423', '190401']  # '190319',,'190306','190320','190321'
# startdays = ['190416','190325','190318','190226','190304','190308']
# startdays = []
# startdays = []
for i in range(1200):
     tmp = 190101+i
     startdays.append(str(tmp))


day = []
# '190102',
# startdays= ['190102','190117','190129','190207']
# startdays.sort()
# startday = '190314'
# startday = startdays[0]

#startdays = ['190813']

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
    file = filenames2.pop()

    vrfile = vr.VitalFile(file)

    for trk in vrfile.trks.values():  # find event track
        # print(trk['name'],tname)
        dname = vrfile.devs[trk['did']]

        if trk['name'].find('ECG_II') >= 0:
            svvflag = 1
            break

#    time_ECG, ECG = vrfile.get_samples('ECG_II')
#    time_ABP, ABP = vrfile.get_samples('ART2')

#    plt.plot(ECG[20000:20000+10000])
#    plt.show()