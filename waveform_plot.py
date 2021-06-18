import vr_reader_fix as vr
import jyLibrary as jy
import vitaldb
import numpy as np
import matplotlib.pyplot as plt


search_date_file=jy.searchDateRoom('J-03',21,2,15)


ART_time,ART_data=jy.findMachineInfo(search_date_file[0],None,'ART2')

length=len(ART_data)//2

plt.figure(figsize=(20, 10))
plt.plot(ART_time[length:length+5000],ART_data[length:length+5000])
plt.show()



