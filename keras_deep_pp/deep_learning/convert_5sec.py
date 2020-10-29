import os
import numpy as np

path = '/home/jmkim/data2/'
output_path = '/home/jmkim/data_5sec/'

files = os.listdir(path)

for file in files:
    if file.find('180628')<0:
        continue
    print(file)

    data = np.load(path +file)
    np.save(output_path + '5sec_'+ file[:-4],data[:5000])
