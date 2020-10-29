import os
import numpy as np
import pandas as pd
import datetime
import pymysql
import matplotlib.pyplot as plt
from utils.processing import *


optimepath = '/home/projects/SVV/SVV_data_191015_optime.xlsx'

wd = pd.read_excel(optimepath,encording='cp949')
dtroom = wd[['수술일자','수술방']]
dtroom =np.array(dtroom)
wd = np.array(wd)
flags = wd[:,4]

op_low = []
op_mid = []
op_high = []


for i in range(len(wd)):


    date = dtroom[i, 0]
    date = str(date.year)[2:] + str(date.month).zfill(2) + str(date.day).zfill(2)
    room = dtroom[i, 1]
    room = room[0] + '-' + room[1:]
    flag = flags[i]




    if flag == 1:
        op_low.append([date,room])
    elif flag == 2:
        op_mid.append([date,room])
    elif flag == 3:
        op_high.append([date,room])



#corr
savepath = '/home/projects/pcg_transform/pcg_AI/deep/SVV/result/'+'svv_paperset_t3/'
filename = 'result.csv'

wd = pd.read_csv(savepath+filename)
corrcsv = np.array(wd)

def matching_flag(opers,corrcsv):
    result = []
    for date,room in opers:
        for csvs in corrcsv:
            if room == csvs[0] and int(date) == csvs[1]:
                result.append(csvs)

    return np.array(result)

result_low = matching_flag(op_low,corrcsv)
result_mid = matching_flag(op_mid,corrcsv)
result_high = matching_flag(op_high,corrcsv)

np.mean(result_low[:,4])
np.mean(result_mid[:,4])
np.mean(result_high[:,4])
"""
import csv
with open('/home/projects/SVV/' + 'reuslt_corr.csv', 'w', newline='') as csvfile:
    wr = csv.writer(csvfile)
    #wr.writerow(['seperated set'])
    wr.writerow(['operation','mae','mse','corr'])
    wr.writerow(['op_low',np.mean(result_low[:,2]),np.mean(result_low[:,3]),np.mean(result_low[:,4])])
    wr.writerow(['op_mid', np.mean(result_mid[:, 2]), np.mean(result_mid[:, 3]), np.mean(result_mid[:, 4])])
    wr.writerow(['op_high', np.mean(result_high[:, 2]), np.mean(result_high[:, 3]), np.mean(result_high[:, 4])])
    wr.writerow(['total', np.mean(corrcsv[:, 2]), np.mean(corrcsv[:, 3]), np.mean(corrcsv[:, 4])])
"""


#bland Altman
savepath = '/home/projects/pcg_transform/pcg_AI/deep/SVV/result/'+'svv_paperset_t3/'
# savepath = '/home/projects/pcg_transform/pcg_AI/deep/SVV/result/'+'svv_paperset_t3_validation/'
#os.listdir(savepath)
filename = 'all_data.csv'

wd = pd.read_csv(savepath + filename)
wd = np.array(wd)

result_all_low = matching_flag(op_low,wd)
result_all_mid = matching_flag(op_mid,wd)
result_all_high = matching_flag(op_high,wd)


all_dataset = []
for i in range(len(wd)):
    if wd[i,0] == 'D-06' and wd[i,1] == 190102:
        #print('190102')
        continue
    if wd[i, 0] == 'D-05' and wd[i, 1] == 190127:
        #print('190127')
        continue
    if wd[i, 1] > 190227:
        #print('190228')
        continue

    all_dataset.append(wd[i])


from scipy import stats

all_dataset = np.array(all_dataset)

test = np.vstack([all_dataset,save_val])


ttt = stats.ttest_ind(all_dataset[:,4],save_val[:,4],equal_var=False)

# result = stats.ttest_ind(Outdoor.Revenue, Department.Revenue, equal_var=False)
print('t statistic : %.3f \np-value : %.3f' % (ttt))

np.mean(test[:,4])
np.std(test[:,4])


np.mean(all_dataset[:,4])
np.std(all_dataset[:,4])

save_val = all_dataset

total_pred = all_dataset[:, 3]
total_svv = all_dataset[:, 4]




def draw_bl(csv,titlename):
    array_bae_mean, array_bae_diff = bland_altman_result(csv[:,3],csv[:,4])
      #limits of agreement
    std = np.std(array_bae_mean)
    min_bae = np.mean(array_bae_mean) - 1.96 * std
    max_bae = np.mean(array_bae_mean) + 1.96 * std


    md = np.mean(array_bae_diff)  # Mean of the difference
    sd = np.std(array_bae_diff)  # Standard deviation of the difference

    cnt = 0
    for i in range(len(csv)):

        if i>0:
            if csv[i,0] != csv[i-1,0] or csv[i,1] != csv[i-1,1]\
                    or i == len(csv):
                cnt = cnt+1


    c_lst = [plt.cm.rainbow(a) for a in np.linspace(0.0, 1.0, cnt)]
    fig = plt.figure(figsize=(20, 10))
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--',alpha=0.5)
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--',alpha=0.5)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)

    print(1)
    cnt = 0
    pred,svv =[],[]
    for i in range(len(csv)):

        if i>0:
            if csv[i,0] != csv[i-1,0] or csv[i,1] != csv[i-1,1]\
                    or i == len(csv):

                bae, diff = bland_altman_result(pred, svv)
                plt.scatter(bae, diff, c=c_lst[cnt], s=0.2)
                cnt = cnt +1
                svv,pred = [],[]

                print(csv[i])
        pred.append(csv[i,3])
        svv.append(csv[i,4])


    #plt.title(titlename)

    #plt.show()
    plt.savefig(
        savepath +titlename+ '.png',dpi=300)
    plt.close()

    return cnt




def draw_concor(csv,titlename):


    #con_rate = np.mean(2 * result_all_low[:, 3] / (result_all_low[:, 3] + result_all_low[:, 4]))
    #con = result_all_mid[:, 4] - result_all_mid[:, 3]
    con = all_dataset[:, 4] - all_dataset[:, 3]
    md = np.mean(con)
    sd = np.std(con)

    cnt = 0
    for i in range(len(csv)):

        if i>0:
            if csv[i,0] != csv[i-1,0] or csv[i,1] != csv[i-1,1]\
                    or i == len(csv):
                cnt = cnt+1


    #fig = plt.figure(figsize=(20, 10))


    c_lst = [plt.cm.rainbow(a) for a in np.linspace(0.0, 1.0, cnt)]
    fig = plt.figure(figsize=(20, 10))
    plt.scatter(all_dataset[:, 3], con, s=0.2)
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')
    print(1)
    cnt = 0
    pred,svv =[],[]
    for i in range(len(csv)):

        if i>0:
            if csv[i,0] != csv[i-1,0] or csv[i,1] != csv[i-1,1]\
                    or i == len(csv):

                bae, diff = bland_altman_result(pred, svv)
                plt.scatter(bae, diff, c=c_lst[cnt], s=0.2)
                cnt = cnt +1
                svv,pred = [],[]

                print(csv[i])
        pred.append(csv[i,3])
        svv.append(csv[i,4])


    #plt.title(titlename)

    #plt.show()
    plt.savefig(
        savepath +titlename+ '.png',dpi=300)
    plt.close()

    return cnt


result_all_low =np.array(result_all_low)
result_all_mid =np.array(result_all_mid)
result_all_high =np.array(result_all_high)

num_low = draw_bl(result_all_low,'t1')
num_mid = draw_bl(result_all_mid,'t2')
num_high = draw_bl(result_all_high,'t3')

result_all_low

np.percentile( result_all_low[:,3],5)
np.percentile( result_all_low[:,3],95)
#concordance rate


array_bae_mean, array_bae_diff = bland_altman_result(result_all_low[:,3],result_all_low[:,4])

import csv
with open('/home/projects/SVV/' + 'reuslt_statistics.csv', 'w', newline='') as csvfile:
    wr = csv.writer(csvfile)
    #wr.writerow(['seperated set'])
    wr.writerow(['operation','patients','count','bias','95%','5%','ci'])
    array_bae_mean, array_bae_diff = bland_altman_result(result_all_low[:, 3], result_all_low[:, 4])
    con_rate = np.mean(2 * result_all_low[:, 3] / (result_all_low[:, 3] + result_all_low[:, 4]))
    wr.writerow(['op_low',len(result_low),len(result_all_low),np.mean(array_bae_mean),np.percentile(array_bae_diff,95),np.percentile(array_bae_diff,5),con_rate])

    array_bae_mean, array_bae_diff = bland_altman_result(result_all_mid[:, 3], result_all_mid[:, 4])
    con_rate = np.mean(2 * result_all_mid[:, 3] / (result_all_mid[:, 3] + result_all_mid[:, 4]))
    wr.writerow(['op_mid',len(result_mid),len(result_all_mid),np.mean(array_bae_mean),np.percentile(array_bae_diff,95),np.percentile(array_bae_diff,5),con_rate])

    array_bae_mean, array_bae_diff = bland_altman_result(result_all_high[:, 3], result_all_high[:, 4])
    con_rate = np.mean(2 * result_all_high[:, 3] / (result_all_high[:, 3] + result_all_high[:, 4]))
    wr.writerow(['op_high',len(result_high),len(result_all_high),np.mean(array_bae_mean),np.percentile(array_bae_diff,95),np.percentile(array_bae_diff,5),con_rate])

    all_dataset = np.array(all_dataset)

    array_bae_mean, array_bae_diff = bland_altman_result(all_dataset[:, 3], all_dataset[:, 4])
    con_rate = np.mean(2 * all_dataset[:, 3] / (all_dataset[:, 3] + all_dataset[:, 4]))
    wr.writerow(['total', np.mean(corrcsv[:, 2]),len(all_dataset),np.mean(array_bae_mean),np.percentile(array_bae_diff,95),np.percentile(array_bae_diff,5),con_rate])
