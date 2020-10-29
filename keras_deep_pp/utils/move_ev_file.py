import os
import pymysql
import numpy as np

#ev1000이 있는 데이터를 옮김
filepath ='/home/shmoon/Preprocessed/all/'
move_filepath=  '/home/projects/pcg_transform/pcg_AI/ml/Data/Preprocessed/VG_SV/'
#os.mkdir(move_filepath)


conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                       db='sv_trend_model', charset='utf8')

curs = conn.cursor()

# Connection으로부터 Cursor 생성
# sql = """SELECT dt,PP,SBP,DBP,HR,CVP,SVV_EV from sv_trend_model.prediction_result as pr LEFT JOIN sv_trend_model.preprocessed_file as pf ON pr.id_preprocessed_file = pf.id;"""
sql = """SELECT file_name from sv_trend_model.prep_file where is_vg =1 ;"""
curs.execute(sql)
row = curs.fetchall()

conn.close()

data = np.array(row)
data = data.flatten()
data[0]
#os.system('ls '+move_filepath)


for i in range(len(data)):
    os.system('cp '+filepath+data[i]+' '+move_filepath)
    print(data[i])

