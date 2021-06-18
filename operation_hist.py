import numpy as np
import pandas as pd


def isNaN(string):
    return string != string

def read_rosette_data():
    path = '/home/jmkim/.jupyter/DB_search/OPER_ROOM_DATE_FROM_2015.csv'
    rosette_data = pd.read_csv(path)
    rosette = np.array(rosette_data["Rosette"])
    #     rosette = (rosette_data["Rosette"])
    begin_date = np.array(pd.to_datetime(rosette_data["Start"]))
    end_date = np.array(pd.to_datetime(rosette_data["End"]))

    for i in range(1, len(rosette)):
        if isNaN(rosette[i]) == True: rosette[i] = rosette[i - 1]

    ind = np.where((pd.isnull(begin_date) == False) & (pd.isnull(end_date) == False))[0]

    return rosette[ind], begin_date[ind], end_date[ind]

#operation info load
rosette,tb_begin,tb_end = read_rosette_data()

print(len(rosette))

a = pd.to_datetime(tb_begin, format='%y-%m-%d %H:%M') # excel
tb_begin_sec = a.to_pydatetime()
a = pd.to_datetime(tb_end, format='%y-%m-%d %H:%M') # excel
tb_end_sec = a.to_pydatetime()