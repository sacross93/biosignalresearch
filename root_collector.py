import numpy as np
import pandas as pd

def isNaN(string):
    return string != string

a=pd.read_excel('./ABP/pc_sample.xlsx',engine='openpyxl')
b=pd.read_excel('./ABP/tablet_sample.xlsx',engine='openpyxl')

b['EV1000'][0].split("/")[2]


pcEV1000=[]
tabletEV1000=[]
pcev=[]
taev=[]
flag=0

for i in a :
    print(i)


for i in range(len(a)) :
    if isNaN(a['ETC'][i]) == False :
        tempEV=a['ETC'][i].split("/")[1]
    else :
        tempEV=''
    # break
    for j in range(len(b)-16) :
        if isNaN(b['ETC'][j]) == False:
            if b['ETC'][j].split("/")[0] != "DI-1120" :
                EVtemp=b['ETC'][j].split("/")[2]
            # print(EVtemp)
        else :
            EVtemp=''
        if EVtemp == tempEV and tempEV == EVtemp and EVtemp != '' and tempEV != '' :
            pcEV1000.append(tempEV)
            tabletEV1000.append(EVtemp)
            flag=1
        # elif EVtemp == '' or tempEV == '' :
        #     break

    if flag == 0 :
        # print(EVtemp)
        pcev.append(tempEV)
        taev.append(EVtemp)



large_len=max(len(pcEV1000),len(tabletEV1000),len(taev),len(pcev))
min_len=min(len(pcEV1000),len(tabletEV1000),len(taev),len(pcev))

for i in range(large_len-min_len) :
    if len(pcEV1000) < large_len :
        pcEV1000.append("")
    if len(tabletEV1000) < large_len :
        tabletEV1000.append("")
    if len(taev) < large_len :
        taev.append("")
    if len(pcev) < large_len :
        pcev.append("")


raw_data={"PCCarescape":pcEV1000,"tabletCarescape":tabletEV1000,"tempPC":pcev,"tempta":taev}
raw_data = pd.DataFrame(raw_data)
raw_data.to_excel(excel_writer='ETC.xlsx')


