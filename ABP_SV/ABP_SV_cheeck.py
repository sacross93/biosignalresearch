import numpy as np
import pandas as pd

address = '/home/projects/pcg_transform/Meeting/jy/sv_npz/'
raw_data = pd.read_excel('sv_vital_ver2.xlsx',engine='openpyxl')



for i in range(len(raw_data)) :

    if raw_data['ABP_count'][i] >= 500 :
        temp_sv=np.load(address+raw_data['Vital_name'][i][:-6]+".npz",allow_pickle=True)

        if len(temp_sv['SV_data']) == int(raw_data['SV_count'][i]) :
            print("True")
        else :
            print("error")
            print(temp_sv['SV'])
            print(raw_data['SV_count'])
            break

