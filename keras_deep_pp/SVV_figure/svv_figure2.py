import os
import numpy as np
import pandas as pd
import datetime
import pymysql
import matplotlib.pyplot as plt
from utils.processing import *


savepath = '/home/projects/pcg_transform/pcg_AI/deep/SVV/result/'+'svv_paperset_t3/'
filename = '3ch_result_svv.csv'

wd = pd.read_csv(savepath+filename)
corrcsv = np.array(wd)

corr =corrcsv[:,4]
corrhist = list()
corrhist.append(len([x for x in corr if x<0.8]))
for i in range(0,20):
    corrhist.append(len([x for x in corr if x>0.8+i*0.01 and x<0.82+i*0.01]))
# corrhist.append(len([x for x in corr if x>0.82 and x<0.84]))
# corrhist.append(len([x for x in corr if x>0.84 and x<0.86]))
# corrhist.append(len([x for x in corr if x>0.86 and x<1.88]))

plt.bar(range(80,101,1),corrhist)
plt.title('Correlation histogram')
plt.show()

offset = 0.2
#mae
mae =corrcsv[:,2]
maehist = list()
maehist.append(len([x for x in mae if x<offset]))
for i in range(0,20):
    maehist.append(len([x for x in mae if x>offset+i*0.2 and x<0.4+i*offset]))
# corrhist.append(len([x for x in corr if x>0.82 and x<0.84]))
# corrhist.append(len([x for x in corr if x>0.84 and x<0.86]))
# corrhist.append(len([x for x in corr if x>0.86 and x<1.88]))

plt.bar(range(0,42,2),maehist)
plt.title('Correlation histogram')
plt.show()


maehist2 = maehist.copy()
for i in range(len(maehist)):
    print(maehist[i]/17)
    if maehist[i]/17 >1.0:
        maehist2[i] = maehist2[i]+4
    elif maehist[i]/17 >0.8:
        maehist2[i] = maehist2[i]+3
    elif maehist[i] / 17 > 0.6:
        maehist2[i] = maehist2[i] + 2
    elif maehist[i]/17 >0.4:
        maehist2[i] = maehist2[i]+1
    elif maehist[i]/17 >0.2:
        maehist2[i] = maehist2[i]+1

corrhist2 = corrhist.copy()
for i in range(len(corrhist)):
    print(corrhist[i]/17)
    if corrhist[i]/17 >1.0:
        corrhist2[i] = corrhist2[i]+4
    elif corrhist[i]/17 >0.8:
        corrhist2[i] = corrhist2[i]+3
    elif corrhist[i] / 17 > 0.6:
        corrhist2[i] = corrhist2[i] + 2
    elif corrhist[i]/17 >0.4:
        corrhist2[i] = corrhist2[i]+1
    elif corrhist[i] / 17 > 0.2:
    corrhist2[i] = corrhist2[i] + 1


#
#
#
# plt.subplot(1,2,1)
# plt.bar(np.arange(0.80,1.01,0.01),corrhist2,width=0.01)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.title('Correlation histogram')
#
# plt.subplot(1,2,2)
# plt.bar(np.arange(0,4.2,0.2),maehist2,width=0.2)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.yticks([0,5,10,15,20])
# plt.title('Mean Absoluted Error histogram')
# plt.show()


import os
import numpy as np
import pandas as pd
import datetime
import pymysql
import matplotlib.pyplot as plt
from utils.processing import *
import matplotlib.dates as mdates

path = '/home/projects/SVV/svv_paperset_t3/'
file_name = 'D-02_190207SVV.csv'
#
# path = '/home/projects/SVV/'
# file_name = 'D-02_181114SVV.csv'

date = 190207
room_name = 'D-02'
wd = pd.read_csv(path + file_name)
time = wd[['time']]
wd = np.array(wd)

val_time = np.array(wd[:,2],dtype='datetime64')
#datetime.datetime(val_time[0])
tmp_time = val_time.astype(datetime.datetime)
val_time = []
for dt in tmp_time:
    val_time.append(dt.time())

val_time = np.array(val_time)



pred = wd[:,3]
svv = wd[:,4]


timehour = []
timeidx = []
for i in range(len(val_time)-1):
    if val_time[i].hour != val_time[i+1].hour:
        if val_time[i + 1].hour%2==0:
            timehour.append(str(val_time[i+1].hour))
            timeidx.append(i+1)

timehour = np.array(timehour)
timeidx = np.array(timeidx)


fig = plt.figure(figsize=(20, 10))
plt.subplot(2,1,1)
#plt.title(room_name + ' ' + str(date))
plt.plot(val_time, pred,  label='CNN Model',alpha=0.8,linewidth=5,color='#DC0000B2')
plt.plot(val_time, svv,  label='Medical Device',alpha=0.8,linewidth=5,color='#4DBBD5B2', linestyle='--')

# for axis in ['top','bottom','left','right']:
#     plt.spines[axis].set_linewidth(2)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)

plt.yticks([5,10,15,20],size='20')
plt.xticks(val_time[timeidx],timehour.flatten(),size='20')
plt.ylabel('SVV(%)',size=20)
plt.legend(fontsize='xx-large',frameon=False)
plt.xlabel('Hours',size='20')
plt.gca().xaxis.set_label_coords(0.97,-0.05)
# plt.show()
# plt.close()
#
# plt.subplot(1,2,2)
# plt.scatter(svv[0::20],pred[0::20], s=20,c='red')
# plt.xlim(0,25)
# plt.ylim(0,25)
# plt.plot([0,25],[0,25],c='black')
# # plt.plot(svv,pred,c='black')
# svv = np.array(svv,np.float)
# pred = np.array(pred,np.float)
# m, b = np.polyfit(svv,pred,1)
# plt.plot(svv,m*svv + b, c='blue')
#
# plt.xlabel(r'$\Delta$SVV$_{\rm EV1000}$ (%)',size=20)
# plt.ylabel(r'$\Delta$SVV$_{\rm Model}$ (%)',size=20)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# # plt.title('Correlation histogram')
# plt.gca().spines['bottom'].set_linewidth(2)
# plt.gca().spines['left'].set_linewidth(2)
# plt.yticks([10,20],size='20')
# plt.xticks([0,10,20],size='20')
# plt.gca().text(20, 13,'Correlation : 0.95', ha='center', va='center', fontsize=20, fontweight='normal',color='blue')
# # plt.show()
# # plt.close()
# #
# plt.subplot(2,1,2)
# residual_error = pred-svv
# plt.plot(val_time,residual_error)
# plt.axhline(0, color='black', ls='-', linewidth=1)
# plt.ylim(-4,4)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['bottom'].set_linewidth(2)
# plt.gca().spines['left'].set_linewidth(2)
# plt.xticks(val_time[timeidx],timehour.flatten(),size='20')
# plt.ylabel('Residual',size='20')
# plt.xlabel('Fitted Value',size='20')
#
# plt.subplot(2,2,3)
# plt.scatter(np.arange(1,len(corr)+1), corr,s=20,c='red')
# # plt.bar(np.arange(0.80,1.01,0.01),corrhist2,width=0.01)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# # plt.title('Correlation histogram')
# plt.gca().spines['bottom'].set_linewidth(2)
# plt.gca().spines['left'].set_linewidth(2)
# plt.yticks(size='20')
# plt.xticks(size='20')
# plt.xlabel('patients',size=20)
# plt.ylabel('correlation',size=20)
# plt.ylim([0.8,1])
#
# plt.subplot(2,2,4)
# plt.scatter(np.arange(1,len(mae)+1), mae,s=20,c='red')
# # plt.bar(np.arange(0.80,1.01,0.01),corrhist2,width=0.01)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# # plt.title('Correlation histogram')
# plt.gca().spines['bottom'].set_linewidth(2)
# plt.gca().spines['left'].set_linewidth(2)
# plt.yticks(size='20')
# plt.xticks(size='20')
# plt.xlabel('patients',size=20)
# plt.ylabel('mean absoluted error',size=20)
#
#
#
#
# plt.subplot(2,2,3)
# plt.scatter(wd[:,0], wd[:,1])
# # plt.bar(np.arange(0.80,1.01,0.01),corrhist2,width=0.01)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# # plt.title('Correlation histogram')
# plt.gca().spines['bottom'].set_linewidth(2)
# plt.gca().spines['left'].set_linewidth(2)
# plt.yticks(size='20')
# plt.xticks(size='20')
# plt.xlabel('Correlation',size=20)
# plt.ylabel('Patients',size=20)
# plt.yticks([10,20])
#
# plt.subplot(2,2,4)
# plt.bar(np.arange(0,4.2,0.2),maehist2,width=0.2)
# # plt.bar(np.arange(0,4.2,0.2),maehist2,width=0.2)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.yticks([10,20])
# # plt.title('Mean Absoluted Error histogram')
# plt.gca().spines['bottom'].set_linewidth(2)
# plt.gca().spines['left'].set_linewidth(2)
# plt.yticks(size='20')
# plt.xticks(size='20')
# plt.xlabel('Mean Absoluted Error',size=20)
#
# plt.show()


# plt.savefig('/home/projects/SVV/result_figure/' + 'result_figure_200702' + '.png',bbox_inches='tight',pad_inches=1.0,dpi=600)
# plt.close()