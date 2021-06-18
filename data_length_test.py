import jyLibrary_fix as jy
from multiprocessing import *
import vitaldb as vital
import pandas as pd
import os


app=['D-01','F-08','D-03','D-04','D-05','D-06','C-03']
# tab_trks=[]
# for i in app :
#     app_list=jy.searchDateRoom(i,21,3)
#     for j in app_list :
#         trks_names=vital.vital_trks(j)
#         print(len(trks_names))
#         tab_trks=np.array(tab_trks,trks_names)

app2=['D-02','F-02','F-01','F-03','F-04','C-04','C-05','C-06','K-01','K-02','K-03','K-04']


def find_trks_name(app_list_para) :
    a=[]
    try :
        tab_trks=vital.vital_trks(app_list_para)

        return tab_trks
    except :
        return a

cpu_count = cpu_count()

for i in app :
    app_list=jy.searchDateRoom(i,21)
    with Pool(cpu_count // 2) as p :
        trks_list=p.map(find_trks_name,app_list)

tab_list=[]

for i in trks_list :
    for j in i :
        tab_list.append(j)

EV1000=[]
Carescape=[]
Primus=[]
Root=[]
etc=[]
Vigilance=[]


for i in set(tab_list) :
    if "EV1000" in i :
        EV1000.append(i)
    elif "Carescape" in i :
        Carescape.append(i)
    elif "Primus" in i :
        Primus.append(i)
    elif "Root" in i or "root" in i :
        Root.append(i)
    elif "Vigilance" in i :
        Vigilance.append(i)
    else :
        etc.append(i)

EV1000=sorted(EV1000)
Carescape=sorted(Carescape)
Primus=sorted(Primus)
Root=sorted(Root)
Vigilance=sorted(Vigilance)
etc=sorted(etc)


large_len=max(len(EV1000),len(Carescape),len(Root),len(Primus),len(etc))
min_len=min(len(EV1000),len(Carescape),len(Root),len(Primus),len(etc))

for i in range(large_len-min_len) :
    if len(EV1000) < large_len :
        EV1000.append("")
    if len(Carescape) < large_len :
        Carescape.append("")
    if len(Primus) < large_len :
        Primus.append("")
    if len(Root) < large_len :
        Root.append("")
    if len(Vigilance) < large_len :
        Vigilance.append("")
    if len(etc) < large_len :
        etc.append("")




raw_data = {'EV1000':EV1000,'Carescape':Carescape,"Primus":Primus,"Root":Root,"Vigilance":Vigilance,"ETC":etc}
raw_data = pd.DataFrame(raw_data)
raw_data.to_excel(excel_writer='tablet_sample9.xlsx')


for i in app2 :
    app_list=jy.searchDateRoom(i,21)
    with Pool(cpu_count // 2) as p :
        trks_list=p.map(find_trks_name,app_list)

tab_list=[]

for i in trks_list :
    for j in i :
        tab_list.append(j)

EV1000=[]
Carescape=[]
Primus=[]
Root=[]
etc=[]
Vigilance=[]


for i in set(tab_list) :
    if "EV1000" in i :
        EV1000.append(i)
    elif "Carescape" in i :
        Carescape.append(i)
    elif "Primus" in i :
        Primus.append(i)
    elif "Root" in i or "root" in i :
        Root.append(i)
    elif "Vigilance" in i :
        Vigilance.append(i)
    else :
        etc.append(i)

EV1000=sorted(EV1000)
Carescape=sorted(Carescape)
Primus=sorted(Primus)
Root=sorted(Root)
Vigilance=sorted(Vigilance)
etc=sorted(etc)


large_len=max(len(EV1000),len(Carescape),len(Root),len(Primus),len(etc))
min_len=min(len(EV1000),len(Carescape),len(Root),len(Primus),len(etc))

for i in range(large_len-min_len) :
    if len(EV1000) < large_len :
        EV1000.append("")
    if len(Carescape) < large_len :
        Carescape.append("")
    if len(Primus) < large_len :
        Primus.append("")
    if len(Root) < large_len :
        Root.append("")
    if len(Vigilance) < large_len :
        Vigilance.append("")
    if len(etc) < large_len :
        etc.append("")




raw_data = {'EV1000':EV1000,'Carescape':Carescape,"Primus":Primus,"Root":Root,"Vigilance":Vigilance,"ETC":etc}
raw_data = pd.DataFrame(raw_data)
raw_data.to_excel(excel_writer='tablet_sample9.xlsx')
