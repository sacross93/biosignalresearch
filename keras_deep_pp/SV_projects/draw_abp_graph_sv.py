import os
import numpy as np
import pandas as pd
import datetime
import pymysql
import matplotlib.pyplot as plt





def abp_svv_draw_graph(filedates):
    #filedate = files[1]
    save_path = '/home/projects/pcg_transform/pcg_AI/deep/SVV/_data_graph/'



    for filedate in filedates:
        print('start : ', filedate)

        conn = pymysql.connect(host='localhost', user='root', password='signal@anes',
                               db='abp_svv_generator', charset='utf8')
        curs = conn.cursor()


        sql = """select file_name,EV_SVV,room_name,date from abp_svv_generator.abp_sv_small where date =%s;"""
        curs.execute(sql, (filedate))
        row = curs.fetchall()

        sql = """select distinct room_name from abp_svv_generator.abp_sv_small where date =%s;"""
        curs.execute(sql, (filedate))
        rooms = curs.fetchall()
        #print(filedate)
        #print(rooms)



        conn.close()
        if len(row) == 0:
            continue

        abp = np.array(row)[:, 0]
        svv = np.array(row)[:, 1]
        room_name = np.array(row)[:, 2]
        date = np.array(row)[:, 3]

        rooms = np.array(rooms)[:,0]


        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        for room in rooms:
            print(room)

            if not os.path.isdir(save_path + "/" +room ):
                os.mkdir(save_path + "/" + room)

            if not os.path.isdir(save_path + "/" +room + "/"+ filedate):
                os.mkdir(save_path + "/" +room + "/"+filedate)

        for i in range(len(abp)//30):
            data = np.load(abp[i*30])
            std = np.std(data[:,0])
            fig = plt.figure(figsize=(20, 10))
            plt.plot(data[:,0])
            plt.title(abp[i*30][-22:]+'std : ' +str(std)[:4])
            plt.savefig(save_path + "/" +room_name[i*30] + "/"+ filedate +'/'+  abp[i*30][-22:-4] + '.png' )
            plt.close()
            print(room_name[i*30] + "_"+ filedate +'_'+'save complete' + str(i*30))



files=[]
for i in range(1100):
    tmp = 190101+i
    files.append(str(tmp))

abp_svv_draw_graph(files)