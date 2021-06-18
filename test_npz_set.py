import vr_reader_fix as vr
import jyLibrary as jy
import vitaldb
import numpy as np

def searchWaveform(data):
    waveForm = ['AWP','ECG_II', 'ART1', 'ART2', 'VOLT', 'IBP1', 'IBP3', 'PLETH', 'ECG1', 'CO2', 'AWP','AUDIO','ABP','FLOW','ECG_I','ECG1']
    if any(data in word for word in waveForm):
        return True
    else:
        return False

para_name=[]

vrfile=jy.searchDateRoom('E-07',21,2,5)
trk_name=vitaldb.vital_trks(vrfile[0])

for i in trk_name :
    parameter_name = i.split('/')
    para_name.append(parameter_name[-1])

result_vr=vr.VitalFile(vrfile[0])
print(result_vr)

dic_true={}
result=[]
for i in para_name :
    wave_time=[]
    wave_data=[]
    number_time=[]
    number_data=[]

    if searchWaveform(i) == True :
        wave_time,wave_data = result_vr.get_samples(i)
    else :
        number_time,number_data = result_vr.get_numbers(i)

    if len(wave_time) > 1 :
        dic_file={'name' : i , 'data' : wave_data , 'time' : wave_time}
    if len(number_time) > 1 :
        dic_file={'name' : i , 'data' : number_data , 'time' : number_time}
    dic_true[i]=dic_file

    result.append(dic_file)
    print(dic_file['name'])
    print(len(result))
    print(result[-1]['name'])

dic_true.keys()
result[0]
np.savez('/home/projects/pcg_transform/jy/E-07_210205.npz', para=result)


test_load = np.load('/home/projects/pcg_transform/jy/E-07_210205.npz',allow_pickle=True)

len(test_load['para'][0]['time'])