import os
import numpy as np
import pandas as pd
import datetime
import pymysql
import matplotlib.pyplot as plt
from utils.processing import *




# list = folder_names = ['result_dpdt','svv_compareset_preprocessing','svv_compareset_dcoffset','svv_compareset_fft']
# folder_names = ['svv_paperset_t3/']
# folder_names = ['result_2ch_dataset_22/']
folder_name = 'svv_paperset_t1'
channel = [3,4]

for folder_name in folder_names:

    cnt = 0
    time_gap = 900
    concor_x = []
    concor_y= []


    #bland Altman
    savepath = '/home/projects/pcg_transform/pcg_AI/deep/SVV/result/'+folder_name +'/'

    filename = 'all_data.csv'

    wd = pd.read_csv(savepath + filename)
    wd = np.array(wd)

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


    all_dataset = np.array(all_dataset)

    total_pred = all_dataset[:, channel[0]]
    total_svv = all_dataset[:, channel[1]]
    dirs = os.listdir(savepath)

    cnt = 0
    for file in dirs:
        if file.find('SVV.csv')<0:
            continue

        cnt = cnt+1
        wd = pd.read_csv(savepath + file)
        wd = np.array(wd)

        pred = wd[:,channel[0]]
        svv = wd[:,channel[1]]

        pred_diff = []
        svv_diff = []
        for i in range((len(pred)-1)//time_gap):
            pred_diff.append(((pred[(i+1)*time_gap]-pred[i*time_gap])/pred[i*time_gap])*100)
            svv_diff.append(((svv[(i+1)*time_gap]-svv[i*time_gap])/svv[i*time_gap])*100)
            #pred_diff.append((pred[(i + 1) * time_gap] - pred[i * time_gap]))
            #svv_diff.append((svv[(i + 1) * time_gap] - svv[i * time_gap]))

        # plt.scatter(svv_diff,pred_diff,c=c_lst[cnt],s=0.2)
        concor_x.append(pred_diff)
        concor_y.append(svv_diff)

    suc, fail = 0,0
    for i in range(len(concor_x)):
        for j in range(len(concor_y[i])):
            if np.abs(concor_x[i][j]) <10 and np.abs(concor_y[i][j])<10:
                continue
            if (concor_x[i][j] >0 and concor_y[i][j] > 0) or  (concor_x[i][j] <0 and concor_y[i][j] < 0):
                suc = suc+1
            else: fail = fail+1

    con_rate = suc / (suc + fail)
    print(con_rate)

    array_bae_mean, array_bae_diff = bland_altman_result(total_pred,total_svv)

    #limits of agreement
    std = np.std(array_bae_mean)
    min_bae = np.mean(array_bae_mean) - 1.96 * std
    max_bae = np.mean(array_bae_mean) + 1.96 * std

    plus = np.percentile( array_bae_diff,5)
    minus = np.percentile( array_bae_diff,95)
    md = np.mean(array_bae_diff)  # Mean of the difference
    sd = np.std(array_bae_diff)  # Standard deviation of the difference

    dirs = os.listdir(savepath)

    fig = plt.figure(figsize=(10, 10))
    c_lst = [plt.cm.rainbow(a) for a in np.linspace(0.0, 1.0, len(dirs)//2)]
    # fig = plt.figure(figsize=(10, 10))
    plt.axhline(md, color='black', linewidth='1.3',alpha=0.4)
    plt.axhline(plus, color='gray',  linestyle='--',alpha=0.4)
    plt.axhline(minus, color='gray', linestyle='--',alpha=0.4)


    plt.yticks(size=10)
    plt.xticks(size=10)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    cnt = 0

    for file in dirs:
        if file.find('SVV.csv')<0:
            continue

        cnt = cnt+1
        wd = pd.read_csv(savepath + file)
        wd = np.array(wd)

        pred = wd[:,channel[0]]
        svv = wd[:,channel[1]]

        bae, diff = bland_altman_result(pred[::10], svv[::10])

        plt.scatter(bae,diff,c=c_lst[cnt],s=0.1,alpha=0.4)

    # plt.title("Total_Bland_Altman plot", size = 10)
    plt.xlim(0, 30)
    plt.ylim(-8,6)
    #plt.show()
    plt.savefig(
        '/home/projects/SVV/' +'ALL_BA2' + '.png',dpi=300)
    plt.close()


    #corr graph
    fig = plt.figure(figsize=(10, 10))
    plt.ylabel('SVV of model')
    plt.xlabel('SVV of EV1000')

    cnt = 0
    for file in dirs:
        if file.find('SVV.csv')<0:
            continue

        cnt = cnt+1
        wd = pd.read_csv(savepath + file)
        wd = np.array(wd)

        pred = wd[:,2]
        svv = wd[:,3]
        plt.scatter(svv[::10], pred[::10],c=c_lst[cnt],s=0.2)
        plt.plot([0,100],[0,100],color='gray',  linestyle='--')


    plt.xlim(0,30)
    plt.ylim(0,30)
    plt.title("Concordance plot")
    plt.savefig( '/home/projects/SVV/' +folder_name+ 'ALL_corr' + '.png',dpi=300)
    plt.close()




    c_lst = [plt.cm.rainbow(a) for a in np.linspace(0.0, 1.0, len(dirs)//2)]

    #four qu
    fig = plt.figure(figsize=(5, 5))
    #plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(0, color='gray', linestyle='--',linewidth=0.4)
    plt.axvline(0, color='gray', linestyle='--',linewidth=0.4)

    cnt = 0
    time_gap = 900
    concor_x = []
    concor_y= []

    for file in dirs:
        if file.find('SVV.csv')<0:
            continue

        cnt = cnt+1
        wd = pd.read_csv(savepath + file)
        wd = np.array(wd)

        pred = wd[:,channel[0]]
        svv = wd[:,channel[1]]

        pred_diff = []
        svv_diff = []
        for i in range((len(pred)-1)//time_gap):
            pred_diff.append(((pred[(i+1)*time_gap]-pred[i*time_gap])/pred[i*time_gap])*100)
            svv_diff.append(((svv[(i+1)*time_gap]-svv[i*time_gap])/svv[i*time_gap])*100)
            #pred_diff.append((pred[(i + 1) * time_gap] - pred[i * time_gap]))
            #svv_diff.append((svv[(i + 1) * time_gap] - svv[i * time_gap]))

        # plt.scatter(svv_diff,pred_diff,c=c_lst[cnt],s=0.2)
        concor_x.append(pred_diff)
        concor_y.append(svv_diff)

    suc, fail = 0,0
    for i in range(len(concor_x)):
        for j in range(len(concor_y[i])):
            if np.abs(concor_x[i][j]) <10 and np.abs(concor_y[i][j])<10:
                continue
            if (concor_x[i][j] >0 and concor_y[i][j] > 0) or  (concor_x[i][j] <0 and concor_y[i][j] < 0):
                suc = suc+1
            else: fail = fail+1

    con_rate = suc / (suc + fail)
    plt.title("Four-quadrant plot" +str(round(con_rate,6)*100))
    plt.ylim(-100,100)
    plt.xlim(-100,100)
    plt.plot([-100,100],[-100,100],color='gray',  linestyle='--',linewidth=0.6)
    plt.plot([-10,10,10,-10,-10],[-10,-10,10,10,-10],color='red',  linestyle='--',linewidth=0.8)
    plt.xlabel('△SVV$_{EV1000} $(%)')
    plt.ylabel('△SVV$_{Model} $(%)')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    #plt.show()
    plt.savefig(
        '/home/projects/SVV/' +folder_name+'four-quardrant_v2' + '.png',dpi=300)
    plt.close()


#correlation 95% CI
wd = pd.read_csv(savepath + 'result.csv')
wd = np.array(wd)

mae = wd[:,2]
mse = wd[:,3]
corr = wd[:,4]

def cut95(data):
    std = np.std(data)
    md = np.mean(data)
    result = []
    for i in range(len(mae)):
        if data[i] > md - std*1.96 and data[i] < md + std*1.96:
            result.append(data[i])
    return result

re_mae = cut95(mae)
re_mse = cut95(mse)
re_corr = cut95(corr)


print(np.average(re_mae))
print(np.average(re_mse))
print(np.average(re_corr))


#concordance rate

concor =  (total_svv)
concor_2 = (total_pred)
con_rate = np.mean(2* total_pred/(total_pred+total_svv))

fig = plt.figure(figsize=(20, 10))
plt.scatter(concor_2,concor)
plt.xlim([-20,20])
plt.ylim([-20,20])

plt.show()
plt.close()


plt.scatter(total_pred,total_svv/total_pred)

plt.figure()
# Hold activation for multiple lines on same graph
#plt.hold('on')
# Set x-axis range
plt.xlim((1,9))
# Set y-axis range
plt.ylim((1,9))
# Draw lines to split quadrants
plt.plot([5,5],[1,9], linewidth=4, color='red' )
plt.plot([1,9],[5,5], linewidth=4, color='red' )
plt.title('Quadrant plot')

plt.show()