import os
import numpy as np
import pandas as pd
import datetime
import pymysql
import matplotlib.pyplot as plt
from utils.processing import *
import matplotlib.dates as mdates

path = '/home/projects/SVV/svv_paperset_t3/'
file_name = 'D-05_190102SVV.csv'
#
# path = '/home/projects/SVV/'
# file_name = 'D-02_181114SVV.csv'

date = 190102
room_name = 'D-05'
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


index_min = np.argmin(wd[:,4])
index_max = np.argmax(wd[:,4])
max_time = wd[index_max,2]
min_time = wd[index_min,2]

index_max2 = 6500 + np.argmax(wd[10000:,4])
# index_max2 = 8000 + np.argmax(wd[10000:,4])
index_min2 = 7000 + np.argmin(wd[7000:8000,4])
max_time2 = wd[index_max2,2]
min_time2 = wd[index_min2,2]





def get_target_data(date,room_name,time):
    conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                           db='abp_svv_generator2', charset='utf8')
    curs = conn.cursor()

    sql = """select file_name from abp_svv_ori3c where date =%s and room_name=%s and time = %s;"""
    curs.execute(sql,(date,room_name,time))
    row = curs.fetchall()

    paper_set = row
    len(row)
    conn.close
    data = np.load(str(row[0][0]))
    return data


max_time = '2019-01-02 15:58:33'
max_data = get_target_data(date,room_name,max_time)

min_data = get_target_data(date,room_name,min_time)
# max_data2 = get_target_data(date,room_name,max_time2)
min_time2 = '2019-01-02 17:51:45'
min_data2 = get_target_data(date,room_name,min_time2)



max_data.shape


# timehour = []
# timeidx = []
# for i in range(len(val_time)-1):
#     if val_time[i].hour != val_time[i+1].hour:
#         if val_time[i + 1].hour%2==0:
#             timehour.append(str(val_time[i+1].hour))
#             timeidx.append(i+1)

cnt = 1
timehour = [0]
timeidx = [0]
for i in range(len(val_time) - 1):
    if (i+1)%1600 ==0:
        timehour.append(cnt)
        timeidx.append(i-1)
        cnt = cnt +1

timehour = np.array(timehour)
timeidx = np.array(timeidx)

#plt.close()


fig = plt.figure(figsize=(20, 20))
plt.subplot(2,1,1)
plt.plot(val_time, pred,  label='CNN Model',alpha=0.8,linewidth=5,color='#0072B5FF')
plt.plot(val_time, svv,  label='Reference',alpha=0.8,linewidth=5,color='#DC0000B2', linestyle='--')
plt.gca().text(0.164, 1.02,'A', ha='center', va='center', transform=plt.gca().transAxes, fontsize=20, fontweight='normal')
plt.gca().text(0.36, 1.02,'B', ha='center', va='center', transform=plt.gca().transAxes, fontsize=20, fontweight='normal')
plt.gca().text(0.647, 1.02,'C', ha='center', va='center', transform=plt.gca().transAxes, fontsize=20, fontweight='normal')
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



plt.axvline(max_time, color='black',  linewidth=10,alpha=0.2)
plt.axvline(min_time, color='black', linewidth=10,alpha=0.2)
# plt.axvline(max_time2, color='black', ls='--', linewidth=1)
plt.axvline(min_time2, color='black',  linewidth=10,alpha=0.2)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# plt.show()

# plt.xticks([i * (len(pred) // 5 - 1) for i in range(6)])
plt.legend(fontsize='xx-large',frameon=False)
plt.xlabel('hours',size='20')
plt.gca().xaxis.set_label_coords(0.97,-0.05)


plt.subplot(8,1,5)
# plt.title('a',loc='left',fontweight="bold", size = 15)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.plot(np.array(min_data[:,0]),'r',linewidth=2)
plt.ylabel('A',size=25,rotation='horizontal',labelpad=13)
plt.yticks(size='17')
plt.xlim(-10,1000)
# plt.ylim(-30, 50)

plt.yticks([])
plt.xticks([])

#line and text
dataset = np.array(min_data[:,0],dtype=float)
x1, x2 = 100,194
minloc = np.min(dataset[x1:x2])
maxloc = np.max(dataset[x1:x2])
maxarg = np.argmax(dataset[x1:x2])
plt.gca().fill_between(np.arange(x1,x2,1), minloc,dataset[x1:x2], color ='#0072B5FF')
plt.plot([x1-17,x1-3],[minloc,minloc],c='black')
plt.plot([x1-17,x1-3],[maxloc,maxloc],c='black')
plt.plot([x1-12,x1-12],[minloc,maxloc],c='black')
quarterloc = np.mean([minloc,maxloc])+np.mean([minloc,maxloc])//2
plt.plot([x1-12,x1-40],[quarterloc,(maxloc)-5], color='black')
plt.gca().text((x1-40)/1000, 0.90,r'PP$_{\rm max}$', ha='center', va='center', transform=plt.gca().transAxes, fontsize=15, fontweight='normal')

plt.plot([x1+maxarg-5,x1+maxarg-15],[quarterloc,maxloc+10], color='black')
plt.gca().text((x1+maxarg-5)/1000, 1.02,'Slope', ha='center', va='center', transform=plt.gca().transAxes, fontsize=15, fontweight='normal')

plt.plot([x1+maxarg+10,x1+maxarg+30],[quarterloc,maxloc-10],color='black')
plt.gca().text((x1+maxarg+30)/1000+0.02,0.83 ,r'SV$_{\rm max}$', ha='center', va='center', transform=plt.gca().transAxes, fontsize=15, fontweight='normal')



x1, x2 = 556,648
minloc = np.min(dataset[x1:x2])
maxloc = np.max(dataset[x1:x2])
maxarg = np.argmax(dataset[x1:x2])
plt.gca().fill_between(np.arange(x1,x2,1), minloc,dataset[x1:x2], color ='#0072B5FF')
plt.plot([x1-17,x1-3],[minloc,minloc],c='black')
plt.plot([x1-17,x1-3],[maxloc,maxloc],c='black')
plt.plot([x1-12,x1-12],[minloc,maxloc],c='black')
quarterloc = np.mean([minloc,maxloc])+np.mean([minloc,maxloc])//2
plt.plot([x1-12,x1-40],[quarterloc,(maxloc)-5], color='black')
plt.gca().text((x1-40)/1000, 0.90,r'PP$_{\rm min}$', ha='center', va='center', transform=plt.gca().transAxes, fontsize=15, fontweight='normal')

plt.plot([x1+maxarg-5,x1+maxarg-15],[quarterloc,maxloc+10], color='black')
plt.gca().text((x1+maxarg-5)/1000, 1.02,'Slope', ha='center', va='center', transform=plt.gca().transAxes, fontsize=15, fontweight='normal')

plt.plot([x1+maxarg+10,x1+maxarg+30],[quarterloc,maxloc-10],color='black')
plt.gca().text((x1+maxarg+30)/1000+0.02,0.78 ,r'SV$_{\rm min}$', ha='center', va='center', transform=plt.gca().transAxes, fontsize=15, fontweight='normal')



# plt.show()
# plt.close()
#
plt.subplot(8,1,6)
# plt.title('b',loc='left',fontweight="bold", size = 15)
plt.plot(np.array(max_data[:,0]),'r',linewidth=2)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
# plt.ylim(-30, 50)
plt.yticks([])
plt.ylabel('B',size=25,rotation='horizontal',labelpad=13)
plt.yticks(size='17')
plt.xlim(-10,1000)
plt.xticks([])




#line and text
dataset = np.array(max_data[:,0],dtype=float)
x1, x2 = 85,165
minloc = np.min(dataset[x1:x2])
maxloc = np.max(dataset[x1:x2])
maxarg = np.argmax(dataset[x1:x2])
plt.gca().fill_between(np.arange(x1,x2,1), minloc,dataset[x1:x2], color ='#0072B5FF')
plt.plot([x1-17,x1-3],[minloc,minloc],c='black')
plt.plot([x1-17,x1-3],[maxloc,maxloc],c='black')
plt.plot([x1-12,x1-12],[minloc,maxloc],c='black')
quarterloc = np.mean([minloc,maxloc])+np.mean([minloc,maxloc])//2
plt.plot([x1-12,x1-40],[quarterloc,(maxloc)-1], color='black')
plt.gca().text((x1-40)/1000+0.01, 0.86,r'PP$_{\rm max}$', ha='center', va='center', transform=plt.gca().transAxes, fontsize=15, fontweight='normal')

plt.plot([x1+maxarg-5,x1+maxarg-15],[quarterloc,maxloc+10], color='black')
plt.gca().text((x1+maxarg-5)/1000, 1.02,'Slope', ha='center', va='center', transform=plt.gca().transAxes, fontsize=15, fontweight='normal')

plt.plot([x1+maxarg+10,x1+maxarg+30],[quarterloc,maxloc-1],color='black')
plt.gca().text((x1+maxarg+30)/1000+0.02,0.83 ,r'SV$_{\rm max}$', ha='center', va='center', transform=plt.gca().transAxes, fontsize=15, fontweight='normal')



x1, x2 = 472,550
minloc = np.min(dataset[x1:x2])
maxloc = np.max(dataset[x1:x2])
maxarg = np.argmax(dataset[x1:x2])
plt.gca().fill_between(np.arange(x1,x2,1), minloc,dataset[x1:x2], color ='#0072B5FF')
plt.plot([x1-17,x1-3],[minloc,minloc],c='black')
plt.plot([x1-17,x1-3],[maxloc,maxloc],c='black')
plt.plot([x1-12,x1-12],[minloc,maxloc],c='black')
quarterloc = np.mean([minloc,maxloc])+np.mean([minloc,maxloc])//2
plt.plot([x1-12,x1-40],[quarterloc,(maxloc)], color='black')
plt.gca().text((x1-40)/1000+0.01, 0.63,r'PP$_{\rm min}$', ha='center', va='center', transform=plt.gca().transAxes, fontsize=15, fontweight='normal')

plt.plot([x1+maxarg-5,x1+maxarg-15],[quarterloc,maxloc+10], color='black')
plt.gca().text((x1+maxarg-5)/1000, 0.81,'Slope', ha='center', va='center', transform=plt.gca().transAxes, fontsize=15, fontweight='normal')

plt.plot([x1+maxarg+10,x1+maxarg+30],[quarterloc,maxloc],color='black')
plt.gca().text((x1+maxarg+30)/1000+0.02,0.63 ,r'SV$_{\rm min}$', ha='center', va='center', transform=plt.gca().transAxes, fontsize=15, fontweight='normal')

#
# plt.subplot(8,1,7)
# # plt.title('c',loc='left',fontweight="bold", size = 15)
# plt.plot(np.array(min_data2[:,0]),'r')
#
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['bottom'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['left'].set_visible(False)
# # plt.ylim(-30, 50)
# plt.yticks([])
# plt.ylabel('C',size=25,rotation='horizontal')
# plt.yticks(size='17')
# plt.xlim(-10,1000)
# plt.xticks([])

plt.subplot(8,1,7)
# plt.title('d',loc='left',fontweight="bold", size = 15)
plt.ylabel('C',size=25,rotation='horizontal',labelpad=13)
# plt.gca().yaxis.set_label_coords(-1,-0.2)
plt.plot(np.array(min_data2[:,0],dtype=float),'r',linewidth=2)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.yticks([])
plt.xticks(np.arange(0, 1100, 200), np.arange(0, 12, 2),size=20)
plt.xlim(-0,1000)
plt.xlabel('seconds',size='20')
plt.gca().xaxis.set_label_coords(0.97,-0.3)
plt.gca().spines['bottom'].set_linewidth(2)


#plt.show()
plt.yticks(size='17')



#line and text
dataset = np.array(min_data2[:,0],dtype=float)
x1, x2 = 118,205
minloc = np.min(dataset[x1:x2])
maxloc = np.max(dataset[x1:x2])
maxarg = np.argmax(dataset[x1:x2])
plt.gca().fill_between(np.arange(x1,x2,1), minloc,dataset[x1:x2], color ='#0072B5FF')
plt.plot([x1-17,x1-3],[minloc,minloc],c='black')
plt.plot([x1-17,x1-3],[maxloc,maxloc],c='black')
plt.plot([x1-12,x1-12],[minloc,maxloc],c='black')
quarterloc = np.mean([minloc,maxloc])+np.mean([minloc,maxloc])//2
plt.plot([x1-12,x1-40],[quarterloc,(maxloc)-5], color='black')
plt.gca().text((x1-40)/1000-0.005, 0.90,r'PP$_{\rm max}$', ha='center', va='center', transform=plt.gca().transAxes, fontsize=15, fontweight='normal')

plt.plot([x1+maxarg-5,x1+maxarg-15],[quarterloc,maxloc+10], color='black')
plt.gca().text((x1+maxarg-5)/1000-0.005, 1.02,'Slope', ha='center', va='center', transform=plt.gca().transAxes, fontsize=15, fontweight='normal')

plt.plot([x1+maxarg+10,x1+maxarg+30],[quarterloc,maxloc-10],color='black')
plt.gca().text((x1+maxarg+30)/1000+0.01,0.83 ,r'SV$_{\rm max}$', ha='center', va='center', transform=plt.gca().transAxes, fontsize=15, fontweight='normal')



x1, x2 = 545,630
minloc = np.min(dataset[x1:x2])
maxloc = np.max(dataset[x1:x2])
maxarg = np.argmax(dataset[x1:x2])
plt.gca().fill_between(np.arange(x1,x2,1), minloc,dataset[x1:x2], color ='#0072B5FF')
plt.plot([x1-17,x1-3],[minloc,minloc],c='black')
plt.plot([x1-17,x1-3],[maxloc,maxloc],c='black')
plt.plot([x1-12,x1-12],[minloc,maxloc],c='black')
quarterloc = np.mean([minloc,maxloc])+np.mean([minloc,maxloc])//2
plt.plot([x1-12,x1-40],[quarterloc,(maxloc)-5], color='black')
plt.gca().text((x1-40)/1000-0.005, 0.85,r'PP$_{\rm min}$', ha='center', va='center', transform=plt.gca().transAxes, fontsize=15, fontweight='normal')

plt.plot([x1+maxarg-5,x1+maxarg-15],[quarterloc,maxloc+10], color='black')
plt.gca().text((x1+maxarg-5)/1000-0.005, 1,'Slope', ha='center', va='center', transform=plt.gca().transAxes, fontsize=15, fontweight='normal')

plt.plot([x1+maxarg+10,x1+maxarg+30],[quarterloc,maxloc-10],color='black')
plt.gca().text((x1+maxarg+30)/1000+0.01,0.80 ,r'SV$_{\rm min}$', ha='center', va='center', transform=plt.gca().transAxes, fontsize=15, fontweight='normal')


plt.savefig('/home/projects/SVV/result_figure/' + 'result_figure_200805' + '.png',bbox_inches='tight',pad_inches=1.0,dpi=600)#
#
# plt.show()
plt.close()
