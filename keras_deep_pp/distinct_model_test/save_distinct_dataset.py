import sklearn
import sklearn.metrics
from utils.my_classes import DataGenerator_npz
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from utils.processing import *
from utils.models import *
import csv

import keras
#from SVV_projects.read_SVV import read_abp_svv, read_abp_svv_10sec,read_abp_svv_minmax_fft
from PCG_SVV.pcg_models import *

#import sklearn.metrics
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D,Dense,LSTM,Dropout
import pymysql
import os
from tcn.tcn import  compiled_tcn
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,3";


#files = [ '180206', '180207', '180208', '180209',  '180213', '180214']

HOSTNAME = 'localhost'
USERNAME = 'root'
PASSWORD = 'signal@anes'
DBNAME = 'abp_svv_generator2'
DEVICE_DB_NAME = 'Vital_DB'


def conver_distinct_PCG_SVV(table_name, get_table_name,type):
    conn = pymysql.connect(host=HOSTNAME, user=USERNAME, password=PASSWORD,
                           db=DBNAME, charset='utf8')
    curs = conn.cursor()

    sql = """select * from data_generator."""+get_table_name+""" where type = %s   order by date,pcg_file;"""
    curs.execute(sql, (type))
    row = curs.fetchall()

    data = np.array(row)

    curs.close()
    conn.close()

    conn = pymysql.connect(host='localhost', user=USERNAME, password=PASSWORD,
                           db=DBNAME, charset='utf8')
    curs = conn.cursor()

    for i in range(len(data) - 1):
        if data[i, -1] == data[i + 1, -1]:
            continue
        else:

            sql = """insert into data_generator."""+table_name+""" select * from data_generator."""+get_table_name+""" where id = %s;"""
            curs.execute(sql, (data[i + 1, 0]))
            row = curs.fetchall()
    conn.commit()

    curs.close()
    conn.close()


def conver_distinct_PCG_SV(table_name, get_table_name,type='minmax'):
    conn = pymysql.connect(host=HOSTNAME, user=USERNAME, password=PASSWORD,
                           db=DBNAME, charset='utf8')
    curs = conn.cursor()

    sql = """select * from """+DBNAME+'.'+get_table_name+""" where type = %s order by date,pcg_file;"""
    curs.execute(sql,(type) )
    row = curs.fetchall()

    data = np.array(row)

    curs.close()
    conn.close()

    conn = pymysql.connect(host='localhost', user=USERNAME, password=PASSWORD,
                           db=DBNAME, charset='utf8')
    curs = conn.cursor()

    for i in range(len(data) - 1):
        if data[i, -1] == data[i + 1, -1]:
            continue
        else:

            sql = """insert into """+DBNAME+'.'+table_name+""" select * from """+DBNAME+'.'+get_table_name+""" where id = %s;"""
            curs.execute(sql, (data[i + 1, 0]))
            row = curs.fetchall()
    conn.commit()

    curs.close()
    conn.close()




def conver_distinct_ABP_SVV(table_name, get_table_name):
    conn = pymysql.connect(host=HOSTNAME, user=USERNAME, password=PASSWORD,
                           db=DBNAME, charset='utf8')
    curs = conn.cursor()

    sql = """select * from """+DBNAME+'.'+get_table_name+""" order by file_name;"""
    curs.execute(sql, )
    row = curs.fetchall()

    data = np.array(row)

    curs.close()
    conn.close()

    conn = pymysql.connect(host='localhost', user=USERNAME, password=PASSWORD,
                           db=DBNAME, charset='utf8')
    curs = conn.cursor()

    for i in range(len(data) - 1):
        if data[i, -1] == data[i + 1, -1]:
            continue
        else:

            sql = """insert into """+DBNAME+'.'+table_name+""" select * from """+DBNAME+'.'+get_table_name+""" where file_name = %s;"""
            curs.execute(sql, (data[i + 1, 4]))
            row = curs.fetchall()
            print(data[i])
    conn.commit()

    curs.close()
    conn.close()
#
# type = 'minmax'
# table_name ='PCG_SVV_distinct'
# get_table_name = 'PCG_SVV'

def PCG_SV_distinct():
    type = 'minmax'
    table_name ='PCG_SV_distinct'
    get_table_name = 'PCG_SV'

    conver_distinct_PCG_SV(table_name,get_table_name);

def ABP_SVV_distinct():
    table_name = 'abp_svv_distinct'
    get_table_name = 'abp_svv_dataset'

    conver_distinct_ABP_SVV(table_name, get_table_name);

def ABP_SV_distinct():
    type = 'minmax'
    table_name = 'abp_svv_distinct'
    get_table_name = 'svv_test1'

    conver_distinct_PCG_SV(table_name, get_table_name);

ABP_SVV_distinct()