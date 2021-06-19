import numpy as np
import pandas as pd
import datetime
import os
import shutil as sh

address="/home/projects/pcg_transform/jy/new_ABP_data/"
move_address="/home/projects/pcg_transform/jy/temp_abp3/"
move_address2="/home/projects/pcg_transform/jy/temp_abp5/"

temp_addr=os.listdir(address)


count1=0
count2=0
for i in temp_addr :
    temp_npz=np.load(address+i,allow_pickle=True)
    temp_time=temp_npz['ABP_time'][-1]-temp_npz['ABP_time'][0]
    if temp_time / 60 / 60 <= 3 and count2 <= 1000 :
        print(temp_time/60/60)
        print(i)
        sh.move(address+i,move_address+i)
        count2+=1
    if count2 > 1000 :
        print("count full")
        break



# for i in temp_addr :
#     temp_npz=np.load(address+i,allow_pickle=True)
#     temp_time=temp_npz['ABP_time'][-1]-temp_npz['ABP_time'][0]
#     if 5 > temp_time / 60 / 60 > 3 and count1 <= 1000 :
#         print(temp_time/60/60)
#         print(i)
#         sh.move(address+i,move_address2+i)
#         count1+=1
#     elif temp_time / 60 / 60 <= 3 and count2 <= 1000 :
#         print(temp_time/60/60)
#         print(i)
#         sh.move(address+i,move_address+i)
#         count2+=1
#     if count1 > 1000 and count2 > 2000 :
#         print("count full")
#         break

