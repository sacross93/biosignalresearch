import pymysql
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import struct
import datetime

DB_SERVER_HOSTNAME = 'localhost'
DB_SERVER_USER = 'jmkim'
DB_SERVER_PASSWORD = 'anesthesia'
DB_SERVER_DATABASE = 'Vital_DB'


def get_compared_track():
    return {'wS12': ['SV', 'PP'], 'S12': ['SV', 'PP'], 'cS12': ['SV', 'PP'],
            'S2_amp': ['SV', 'PP', 'SVR'], 'cS2_amp': ['SV', 'PP', 'SVR'], 'wS2_amp': ['SV', 'PP', 'SVR'],
            'S1_amp': ['SV', 'PP', 'SVR'], 'cS1_amp': ['SV', 'PP', 'SVR'], 'wS1_amp': ['SV', 'PP', 'SVR'],
            'S2_Area_amp': ['SV', 'PP', 'SVR'], 'S1_Area_amp': ['SV', 'PP', 'SVR'],
            'S12_itvV': ['PPV', 'SVV'], 'wS12_itvV': ['PPV', 'SVV'], 'S1ampV': ['PPV', 'SVV'],
            'S2ampV': ['PPV', 'SVV'], 'S1_Area_ampV': ['PPV', 'SVV'], 'S2_Area_ampV': ['PPV', 'SVV'],

            }


# RS2,RRint,S21int,wS21int,per_s12,per_s21, per_wS12, wper_wS21
# RS2V,RRintV,S21intV,wS21intV
def get_compared_track_v2():
    return {'wS12': ['SV', 'PP'], 'S12': ['SV', 'PP'], 'cS12': ['SV', 'PP'],
            'S2_amp': ['SV', 'PP', 'SVR'], 'cS2_amp': ['SV', 'PP', 'SVR'], 'wS2_amp': ['SV', 'PP', 'SVR'],
            'S1_amp': ['SV', 'PP', 'SVR'], 'cS1_amp': ['SV', 'PP', 'SVR'], 'wS1_amp': ['SV', 'PP', 'SVR'],
            'S2_Area_amp': ['SV', 'PP', 'SVR'], 'S1_Area_amp': ['SV', 'PP', 'SVR'],
            'S12_itvV': ['PPV', 'SVV'], 'wS12_itvV': ['PPV', 'SVV'], 'S1ampV': ['PPV', 'SVV'],
            'S2ampV': ['PPV', 'SVV'], 'S1_Area_ampV': ['PPV', 'SVV'], 'S2_Area_ampV': ['PPV', 'SVV'],
            'RS2' : ['SV','PP'], 'RRint' : ['SV','PP','SVR'],
            'S21int' : ['SV','PP','SVR'], 'wS21int' : ['SV','PP','SVR'],
            'per_S12': ['SV', 'PP', 'SVR'], 'per_S21': ['SV', 'PP', 'SVR'],
            'per_wS12': ['SV', 'PP', 'SVR'], 'per_wS21': ['SV', 'PP', 'SVR'],
            'RS2V': ['PPV', 'SVV'], 'RRintV': ['PPV', 'SVV'], 'S21intV': ['PPV', 'SVV'],'wS21intV': ['PPV', 'SVV']
            }


def get_parameter_list():
    return ['PPV_S12_itvV', 'PPV_S1_Area_ampV', 'PPV_S1ampV', 'PPV_S2_Area_ampV', 'PPV_S2ampV', 'PPV_wS12_itvV',
            'PP_S12', 'PP_S1_Area_amp', 'PP_S1_amp', 'PP_S2_Area_amp', 'PP_S2_amp', 'PP_cS12', 'PP_cS1_amp',
            'PP_cS2_amp', 'PP_wS12',
            'PP_wS1_amp', 'PP_wS2_amp', 'SVR_S1_Area_amp', 'SVR_S1_Area_amp_10min', 'SVR_S1_amp',
            'SVR_S1_amp_10min', 'SVR_S2_Area_amp',
            'SVR_S2_Area_amp_10min', 'SVR_S2_amp', 'SVR_S2_amp_10min', 'SVR_cS1_amp', 'SVR_cS1_amp_10min',
            'SVR_cS2_amp', 'SVR_cS2_amp_10min',
            'SVR_wS1_amp', 'SVR_wS1_amp_10min', 'SVR_wS2_amp', 'SVR_wS2_amp_10min', 'SVV_S12_itvV',
            'SVV_S1_Area_ampV', 'SVV_S1ampV', 'SVV_S2_Area_ampV',
            'SVV_S2ampV', 'SVV_wS12_itvV', 'SV_S12', 'SV_S12_10min', 'SV_S1_Area_amp', 'SV_S1_Area_amp_10min',
            'SV_S1_amp', 'SV_S1_amp_10min',
            'SV_S2_Area_amp', 'SV_S2_Area_amp_10min', 'SV_S2_amp', 'SV_S2_amp_10min', 'SV_cS12', 'SV_cS12_10min',
            'SV_cS1_amp', 'SV_cS1_amp_10min',
            'SV_cS2_amp', 'SV_cS2_amp_10min', 'SV_wS12', 'SV_wS12_10min', 'SV_wS1_amp', 'SV_wS1_amp_10min',
            'SV_wS2_amp', 'SV_wS2_amp_10min']


def get_parameter_list_v2():

    datanames = []
    labels = get_compared_track_v2()

    for key in labels.keys():
        #print(labels[key])
        for name in labels[key]:
            tmp_name = name+'_' + key
            datanames.append(tmp_name)

    return datanames



def get_gzip_size(filename):
    with open(filename, 'rb') as f:
        f.seek(-4, 2)
        data = f.read(4)
    size = struct.unpack('<L', data)[0]
    return size


# def db_upload(filepath, )

def device_check(vrfile):
    devices = []
    for cnt in range(len(vrfile.devs)):
        did = list(vrfile.devs.keys())[cnt]
        if did != 0:
            devices.append(vrfile.devs[did]['name'])

    return devices


def get_labels(device):
    labels = {'Bx50': ['ST', 'ST_I', 'ST_II', 'ST_III', 'ST_V', 'ST_AVL', 'ST_AVR', 'ST_AVF', 'HR', 'PVC', 'SVO2',
                       'PLETH_SPO2', 'PLETH_HR', 'RR',
                       'ETCO2', 'INCO2', 'RR_CO2', 'AMB_PRES', 'FIO2', 'FEO2', 'NIBP_SBP', 'NIBP_DBP', 'NIBP_MBP',
                       'MIBP_HR', 'ART_SBP', 'ART_MBP',
                       'ART_DBP', 'ART_HR', 'FEM_SBP', 'FEM_MBP', 'FEM_DBP', 'FEM_HR', 'PA_SBP', 'PA_DBP', 'PA_HR',
                       'LAP', 'RAP', 'CVP', 'BT', 'N20_ET',
                       'N20_IN', 'AGENT_ET', 'AGENT_IN', 'AGENT_MAC', 'RR_VENT', 'PPEAK', 'PEEP', 'PPLAT', 'MV',
                       'TV_INSP', 'TV_EXP',
                       'COMPLIANCE'],
              'Intellivue': ['CO2_ET', 'CO2_INSP_MIN', 'AWAY_RR', 'AWAY_TOT', 'AWAY_O2_INSP', 'ISOFL_ET', 'ISOFL_INSP',
                             "N2O_ET",
                             'N2O_INSP', 'O2_ET', 'O2_INSP', 'SEVOFL_ET', 'SEVOFL_INSP', 'ART_SYS', 'ART_DIA',
                             'ART_MEAN', 'HR',
                             'TEMP', 'TEMP_ESOPH', 'PLETH_HR', 'PERF_REL', 'PLETH_SAT_O2', 'ECG_HR', 'RR', 'BIS_BIS',
                             'BIS_EMG',
                             'BIS_SQI', 'BIS_SEF', 'BIS_SR', 'ABP_SYS', 'ABP_DIA', 'ABP_MEAN', 'PPV', 'CVP_MEAN',
                             'PAP_SYS',
                             'PAP_DIA', 'PAP_MEAN', 'NIBP_HR', 'NIBP_SYS', 'NIBP_DIA', 'NIBP_MEAN', 'ST_I', 'ST_II',
                             'ST_III',
                             'ST_MCL', 'ST_V', 'ST_AVF', 'ST_AVL', 'ST_AVR', 'QT_GL', 'QT_HR', 'QTc', 'QTc_DELTA',
                             'DESFL_INSP',
                             'DESFL_ET', 'SET_SPEEP', 'SET_INSP_TIME', 'PLAT_TIME', 'RISE_TIME', 'TV_IN', 'SV', 'SI',
                             'REF',
                             'ICP_MEAN', 'TOF_RATIO', 'TOF_CNT', 'TOF_1', 'TOF_2', 'TOF_3', 'TOF_4'],
              'BIS': ['SQI', 'EMG', 'SR', 'SEF', 'BIS', 'TOTPOW'],
              # 'Invos': ['SCO2_L','SCO2_R','SCO2_S1','SCO2_S2'],
              'CardioQ': ['CO', 'SV', 'HR', 'MD', 'SD', 'FTc', 'Ftp', 'MA', 'PV', 'CI', 'SVI'],
              # 'Vigileo':['CO','CI','SV','SVI','SVV'],
              'EV1000': ['CVP', 'SVR', 'SVRI', 'ART_MBP', 'CO', 'CI', 'SV', 'SVI', 'SVV'],
              'Vigilance': ['HR_AVR', 'BT_PA', 'SQI', 'SNR', 'SVO2', 'CO', 'CI', 'SV', 'SVI', 'SVR', 'SVRI', 'EDV',
                            'EDVI', 'ESV', 'ESVI', 'RVEF'],
              # 'Orchestra':['PUMP1_DRUG','PUMP1_CONC','PUMP1_RATE','PUMP1_VOL','PUMP1_REMAIN','PUMP1_PRES','PUMP1_CP','PUMP1_CE','PUMP1_CT',
              #             'PUMP2_DRUG','PUMP2_CONC','PUMP2_RATE','PUMP2_VOL','PUMP2_REMAIN','PUMP2_PRES','PUMP2_CP','PUMP2_CE','PUMP2_CT'],
              # 'RI-2':['FLOW_RATE','INPUT_TEMP','OUTPUT_TEMP','INPUT_AMB_TEMP','TOTAL_VOL','PRESSURE'],
              }

    for key in list(labels.keys()):
        if device == key:
            return labels[device]


def get_db_table(device):
    tables = {'Intellivue': 'number_ph_vital',
              'Bx50': 'number_ge_vital',
              'BIS': 'number_bs_vital',
              'CardioQ': 'number_cdq_vital',
              'EV1000': 'number_ev_vital',
              'Vigilance': 'number_vg_vital'
              }

    return tables[device]


def insert_to_DB(room, dname, raw_data):
    timestamp_interval = 0.5
    timestamp = 0
    trackdata = {}
    table_name = get_db_table(dname)
    rosette = room[:-3]
    bed = room[-2:]

    for i, dataset in enumerate(raw_data):
        # print(dataset)
        if timestamp != 0 and (dataset[1] > timestamp + 0.5 or len(raw_data) - 1 == i):
            # print(timestamp)

            if len(trackdata) == 0:
                continue
            # print('start save')

            tracks = list(trackdata.keys())
            tracks_str = ''
            tracks_value = [1, rosette, bed, datetime.datetime.utcfromtimestamp(timestamp + 3600 * 9)]
            track_count = '%s,%s,%s,%s'
            for t in tracks:
                tracks_str = tracks_str + ',' + t
                tracks_value.append(trackdata[t])
                track_count = track_count + ',%s'

            conn = pymysql.connect(host=DB_SERVER_HOSTNAME, user=DB_SERVER_USER, password=DB_SERVER_PASSWORD,
                                   db=DB_SERVER_DATABASE, charset='utf8')
            curs = conn.cursor()

            # sql = """SELECT dt,PP,SBP,DBP,HR,CVP,SVV_EV from sv_trend_model.prediction_result as pr LEFT JOIN sv_trend_model.preprocessed_file as pf ON pr.id_preprocessed_file = pf.id;"""
            sql = 'insert into ' + table_name + '(method,rosette,bed,dt' + tracks_str + ')' + \
                  'values (' + track_count + ');'
            # print(sql)
            # print(tracks_value)
            curs.execute(sql, (tracks_value))
            conn.commit()
            conn.close()

            trackdata = {}
            timestamp = 0

        if timestamp == 0:
            timestamp = dataset[1]

        track = {dataset[2]: dataset[3]}
        trackdata.update(track)


HOSTNAME = 'localhost'
USERNAME = 'jmkim'
PASSWORD = 'anesthesia'
DBNAME = 'data_generator'
DEVICE_DB_NAME = 'Vital_DB'


def load_pcg_data(room, date):
    row = []
    conn = pymysql.connect(host=HOSTNAME, user=USERNAME, password=PASSWORD,
                           db=DBNAME, charset='utf8')
    curs = conn.cursor()

    sql = "select time,file_name from pcg_abp_features where date = %s and room_name=%s order by time ;"
    curs.execute(sql, (date, room))
    row = curs.fetchall()
    row = np.array(row)

    conn.close()

    if len(row) == 0:
        return [], []

    PCG = np.array(row[:, 1])
    time = np.array(row[:, 0])
    return time, PCG


def load_pcg_data_v2(room, date):
    row = []
    conn = pymysql.connect(host=HOSTNAME, user=USERNAME, password=PASSWORD,
                           db=DBNAME, charset='utf8')
    curs = conn.cursor()

    sql = "select time,file_name from pcg_abp_features_v2 where date = %s and room_name=%s order by time ;"
    curs.execute(sql, (date, room))
    row = curs.fetchall()
    row = np.array(row)

    conn.close()

    if len(row) == 0:
        return [], []

    PCG = np.array(row[:, 1])
    time = np.array(row[:, 0])
    return time, PCG


def load_pcg_feature(room, date, pcg_feature):
    row = []
    conn = pymysql.connect(host=HOSTNAME, user=USERNAME, password=PASSWORD,
                           db=DBNAME, charset='utf8')
    curs = conn.cursor()

    sql = "select time," + pcg_feature + " from pcg_abp_features where date = %s and room_name=%s  order by time ;"
    curs.execute(sql, (date, room))
    row = curs.fetchall()
    row = np.array(row)

    conn.close()

    if len(row) == 0:
        return [], []
    time = row[:, 0]
    PCG = np.array(row[:, 1], dtype=np.float16)
    return time, PCG


def load_pcg_feature_v2(room, date, pcg_feature):
    row = []
    conn = pymysql.connect(host=HOSTNAME, user=USERNAME, password=PASSWORD,
                           db=DBNAME, charset='utf8')
    curs = conn.cursor()

    sql = "select time," + pcg_feature + '_v2' + " from pcg_abp_features where date = %s and room_name=%s  order by time ;"
    curs.execute(sql, (date, room))
    row = curs.fetchall()
    row = np.array(row)

    conn.close()

    if len(row) == 0:
        return [], []
    time = row[:, 0]
    PCG = np.array(row[:, 1], dtype=np.float16)
    return time, PCG

def load_pcg_feature_files(room, date, pcg_feature):
    row = []
    conn = pymysql.connect(host=HOSTNAME, user=USERNAME, password=PASSWORD,
                           db=DBNAME, charset='utf8')
    curs = conn.cursor()

    sql = "select time," + pcg_feature + " from pcg_abp_features where date = %s and room_name=%s  order by time ;"
    curs.execute(sql, (date, room))
    row = curs.fetchall()
    row = np.array(row)

    conn.close()

    if len(row) == 0:
        return [], []
    time = row[:, 0]
    PCG = row[:, 1]
    return time, PCG

def load_pcg_feature_files_v2(room, date, pcg_feature):
    row = []
    conn = pymysql.connect(host=HOSTNAME, user=USERNAME, password=PASSWORD,
                           db=DBNAME, charset='utf8')
    curs = conn.cursor()

    sql = "select time," + pcg_feature + " from pcg_abp_features_v2 where date = %s and room_name=%s  order by time ;"
    curs.execute(sql, (date, room))
    row = curs.fetchall()
    row = np.array(row)

    conn.close()

    if len(row) == 0:
        return [], []
    time = row[:, 0]
    PCG = row[:, 1]
    return time, PCG


def state_check(room, date):
    row = []
    conn = pymysql.connect(host=HOSTNAME, user=USERNAME, password=PASSWORD,
                           db=DBNAME, charset='utf8')
    curs = conn.cursor()

    sql = "select S1_amp,S2_amp from pcg_abp_features where date = %s and room_name=%s  order by time ;"
    curs.execute(sql, (date, room))
    row = curs.fetchall()
    row = np.array(row)

    conn.close()

    if len(row) == 0:
        return 'problem_data'

    S1_amp = np.array(row[:, 0], dtype=np.float16)
    S2_amp = np.array(row[:, 1], dtype=np.float16)

    med_s1 = np.median(S1_amp)
    med_s2 = np.median(S2_amp)

    if med_s2 < 1 and med_s1 < 1:
        return 'problem_data'

    elif med_s1 / med_s2 > 2:
        return 'S1_tall'
    elif med_s2 / med_s1 > 2:
        return 'S2_tall'
    elif med_s1 / med_s2 < 2 and med_s2 / med_s1 < 2:
        return 'normal'
    else:
        return 'problem_data'

    return ''


def state_check_v2(room, date):
    row = []
    conn = pymysql.connect(host=HOSTNAME, user=USERNAME, password=PASSWORD,
                           db=DBNAME, charset='utf8')
    curs = conn.cursor()

    sql = "select S1_amp,S2_amp from pcg_abp_features where date = %s and room_name=%s  order by time ;"
    curs.execute(sql, (date, room))
    row = curs.fetchall()
    row = np.array(row)

    conn.close()

    if len(row) == 0:
        return 'problem_data'

    S1_amp = np.array(row[:, 0], dtype=np.float16)
    S2_amp = np.array(row[:, 1], dtype=np.float16)

    med_s1 = np.median(S1_amp)
    med_s2 = np.median(S2_amp)

    if med_s2 < 1 and med_s1 < 1:
        return 'problem_data'

    elif med_s1 / med_s2 > 2:
        return 'S1_tall'
    elif med_s2 / med_s1 > 2:
        return 'S2_tall'
    elif med_s1 / med_s2 < 2 and med_s2 / med_s1 < 2:
        return 'normal'
    else:
        return 'problem_data'

    return ''

def load_pcg_abp_feature(room, date, pcg_feature, abp_feature):
    row = []
    conn = pymysql.connect(host=HOSTNAME, user=USERNAME, password=PASSWORD,
                           db=DBNAME, charset='utf8')
    curs = conn.cursor()

    sql = "select time," + pcg_feature + "," + abp_feature + " from pcg_abp_features where date = %s and room_name=%s  order by time ;"
    curs.execute(sql, (date, room))
    row = curs.fetchall()
    row = np.array(row)

    conn.close()

    if len(row) == 0:
        return [], [], []

    time = row[:, 0]
    PCG = np.array(row[:, 1], dtype=np.float16)

    ABP = np.array(row[:, 2], dtype=np.float16)

    return time, PCG, ABP


def load_pcg_abp_feature_v2(room, date, pcg_feature, abp_feature):
    row = []
    conn = pymysql.connect(host=HOSTNAME, user=USERNAME, password=PASSWORD,
                           db=DBNAME, charset='utf8')
    curs = conn.cursor()

    sql = "select time," + pcg_feature + "," + abp_feature + " from pcg_abp_features_v2 where date = %s and room_name=%s  order by time ;"
    curs.execute(sql, (date, room))
    row = curs.fetchall()
    row = np.array(row)

    conn.close()

    if len(row) == 0:
        return [], [], []

    time = row[:, 0]
    PCG = np.array(row[:, 1], dtype=np.float16)

    ABP = np.array(row[:, 2], dtype=np.float16)

    return time, PCG, ABP

def svr_graph(room, date):
    row = []
    table = 'number_vg_vital'
    conn = pymysql.connect(host=HOSTNAME, user=USERNAME, password=PASSWORD,
                           db=DBNAME, charset='utf8')
    curs = conn.cursor()

    sql = "select dt,SVR from " + table + " where " + SVR + " is not null and dt>%s and dt<%s and rosette=%s and bed = %s ;"
    curs.execute(sql, (time[0], time[-1], room[:-3], room[-2:]))
    row = curs.fetchall()
    row = np.array(row)

    conn.close()

    if len(row) == 0:
        return [], []

    time = np.array(row[:, 0])
    svr = np.array(row[:,1])

    svr_flag = []
    for i in range(len(svr)):
        if svr>=800:
            svr_flag.append(1)
        else:
            svr_flag.append(0)
    return time, svr_flag


# time ,S2_amp, PP = load_pcg_abp_feature(190201,'S2_amp','PP')


def draw_corr_graph(time, data1, data2, data1_name='data1', data2_name='data2', filter_size=0, delay=0):
    if filter_size % 2 == 0:
        filter_size = filter_size + 1

    if filter_size > 2:
        data1 = scipy.signal.savgol_filter(data1, filter_size, 0)
        data2 = scipy.signal.savgol_filter(data2, filter_size, 0)

    fig = plt.figure(figsize=(20, 10))
    ax2 = fig.subplots()
    # interval and pp corr
    # interval and pp corr
    ax2.plot(time + datetime.timedelta(minutes=delay), data1, marker='s', label=data1_name)
    cor = np.corrcoef(data1, data2)
    stcor = str(round(cor[0, 1], 2))
    plt.legend(loc='upper left')
    ax2 = ax2.twinx()
    ax2.plot(time, data2, 'r', marker='s', label=data2_name)
    plt.legend(loc='upper right')
    plt.title(", correlation:" + stcor)
    plt.show()
    plt.close()


# draw_corr_graph(time,S2_amp,PP,'S2_amp,'PP',17)


def draw_corr_graph_save(time, data1, data2, data1_name='data1', data2_name='data2', savepath='', filter_size=0,
                         delay=0):
    if filter_size % 2 == 0:
        filter_size = filter_size + 1

    if filter_size > 2:
        data1 = scipy.signal.savgol_filter(data1, filter_size, 0)
        data2 = scipy.signal.savgol_filter(data2, filter_size, 0)

    fig = plt.figure(figsize=(20, 10))
    ax2 = fig.subplots()
    ax2.plot(np.array(time) + datetime.timedelta(minutes=delay), data1, marker='s', label=data1_name)
    cor = np.corrcoef(data1, data2)
    stcor = str(round(cor[0, 1], 2))
    plt.legend(loc='upper left')
    ax2 = ax2.twinx()
    ax2.plot(time, data2, 'r', marker='s', label=data2_name)
    plt.legend(loc='upper right')
    plt.title(", correlation:" + stcor)
    plt.savefig(savepath)
    plt.close()

    return stcor


def corr_filtered(time, data1, data2, filter_size=0):
    if filter_size % 2 == 0:
        filter_size = filter_size + 1

    if filter_size > 2:
        data1 = scipy.signal.savgol_filter(data1, filter_size, 0)
        data2 = scipy.signal.savgol_filter(data2, filter_size, 0)
    cor = np.corrcoef(data1, data2)
    stcor = str(round(cor[0, 1], 2))
    return stcor


# draw_corr_graph_save(time,S2_amp,PP,'S2_amp','PP','test_code/test.png',9)
# os.listdir('test_code')

#get pcg 10 sec and svv every 2 sec
def load_matching_device_pcg10svv(room, time, pcg_feature, device, track):
    table = get_db_table(device)

    conn = pymysql.connect(host=HOSTNAME, user=USERNAME, password=PASSWORD,
                           db=DEVICE_DB_NAME, charset='utf8')
    curs = conn.cursor()
    sql = "select dt," + track + " from " + table + " where " + track + " is not null and dt>%s and dt<%s and rosette=%s and bed = %s ;"
    curs.execute(sql, (time[0], time[-1], room[:-3], room[-2:]))
    row = curs.fetchall()
    conn.close()

    # print(len(time))
    # print(len(pcg_feature))

    row = np.array(row)
    if len(row) == 0:
        return [], [], []

    device_time = row[:, 0]
    dev_data = np.array(row[:, 1], dtype=np.float16)

    result_pcg_data = []

    result_data, result_time = [], []
    svv_index = []
    total_cnt = 0
    for i, timeset in enumerate(time):
        tmp_data = []

        for cnt in range(total_cnt, len(device_time)):
            if timeset + datetime.timedelta(seconds=20) > device_time[cnt]:
                if timeset < device_time[cnt]:
                    tmp_data.append(dev_data[cnt])



            else:
                if not tmp_data:
                    break
                svv_index.append([total_cnt,cnt-1,i])#0~ cnt(out of range)-1 , i
                result_time.append(timeset)
                result_pcg_data.append(pcg_feature[i])
                # print(tmp_data)
                result_data.append(np.median(tmp_data))
                total_cnt = cnt
                break

    return result_time, result_pcg_data, svv_index,row[:,0],row[:,1]



def load_matching_device(room, time, pcg_feature, device, track):
    table = get_db_table(device)

    conn = pymysql.connect(host=HOSTNAME, user=USERNAME, password=PASSWORD,
                           db=DEVICE_DB_NAME, charset='utf8')
    curs = conn.cursor()
    sql = "select dt," + track + " from " + table + " where " + track + " is not null and dt>%s and dt<%s and rosette=%s and bed = %s ;"
    curs.execute(sql, (time[0], time[-1], room[:-3], room[-2:]))
    row = curs.fetchall()
    conn.close()

    # print(len(time))
    # print(len(pcg_feature))

    row = np.array(row)
    if len(row) == 0:
        return [], [], []

    device_time = row[:, 0]
    dev_data = np.array(row[:, 1], dtype=np.float16)

    result_pcg_data = []

    result_data, result_time = [], []
    total_cnt = 0
    for i, timeset in enumerate(time):
        tmp_data = []

        for cnt in range(total_cnt, len(device_time)):
            if timeset + datetime.timedelta(seconds=20) > device_time[cnt]:
                if timeset < device_time[cnt]:
                    tmp_data.append(dev_data[cnt])


            else:
                if not tmp_data:
                    break
                result_time.append(timeset)
                result_pcg_data.append(pcg_feature[i])
                # print(tmp_data)
                result_data.append(np.median(tmp_data))
                total_cnt = cnt
                break

    return result_time, result_pcg_data, result_data


# draw_corr_graph(svtime,pcg_sv,sv,'pcg_sv','sv',17)

def load_matching_device_10min_deay_backup(room, time, pcg_feature, device, track):
    table = get_db_table(device)

    conn = pymysql.connect(host=HOSTNAME, user=USERNAME, password=PASSWORD,
                           db=DEVICE_DB_NAME, charset='utf8')
    curs = conn.cursor()
    sql = "select dt," + track + " from " + table + " where " + track + " is not null and dt>%s and dt<%s and rosette=%s and bed = %s ;"
    curs.execute(sql, (time[0], time[-1], room[:-3], room[-2:]))
    row = curs.fetchall()
    conn.close()

    # print(len(time))
    # print(len(pcg_feature))

    row = np.array(row)
    if len(row) == 0:
        return [], [], []

    device_time = row[:, 0]
    dev_data = np.array(row[:, 1], dtype=np.float16)

    result_pcg_data = []

    result_data, result_time = [], []
    total_cnt = 0
    for i, timeset in enumerate(time):
        tmp_data = []

        for cnt in range(total_cnt, len(device_time)):
            if timeset + datetime.timedelta(seconds=20) > device_time[cnt] - datetime.timedelta(minutes=10):
                if timeset < device_time[cnt] - datetime.timedelta(minutes=10):
                    tmp_data.append(dev_data[cnt])


            else:
                if not tmp_data:
                    break
                result_time.append(timeset)
                result_pcg_data.append(pcg_feature[i])
                # print(tmp_data)
                result_data.append(np.median(tmp_data))
                total_cnt = cnt
                break

    return result_time, result_pcg_data, result_data
# svtime, pcg_sv, sv = load_matching_device(time,S2_amp,'Vigilance','SV')

# draw_corr_graph(svtime,pcg_sv,sv,'pcg_sv','sv',17)


def load_matching_device_10min(room, time, pcg_feature, device, track):
    table = get_db_table(device)

    conn = pymysql.connect(host=HOSTNAME, user=USERNAME, password=PASSWORD,
                           db=DEVICE_DB_NAME, charset='utf8')
    curs = conn.cursor()
    sql = "select dt," + track + " from " + table + " where " + track + " is not null and dt>%s and dt<%s and rosette=%s and bed = %s ;"
    curs.execute(sql, (time[0], time[-1], room[:-3], room[-2:]))
    row = curs.fetchall()
    conn.close()

    # print(len(time))
    # print(len(pcg_feature))

    row = np.array(row)
    if len(row) == 0:
        return [], [], []

    device_time = row[:, 0]
    dev_data = np.array(row[:, 1], dtype=np.float16)

    result_pcg_data = []

    result_data, result_time = [], []
    total_cnt = 0


    for i, timeset in enumerate(device_time):
        tmp_data = []

        for cnt in range(total_cnt, len(time)):

            if timeset > time[cnt]:
                if timeset - datetime.timedelta(minutes=10) < time[cnt]:
                    tmp_data.append(pcg_feature[cnt])
                else:
                    total_cnt = cnt



            else:
                if not tmp_data:
                    break
                result_time.append(timeset)
                result_data.append(dev_data[i])
                # print(tmp_data)
                result_pcg_data.append(np.mean(tmp_data))

                break

    return result_time, result_pcg_data, result_data


# 10 min PCG data average and 5min delay
def load_matching_device_10min_5delay(room, time, pcg_feature, device, track):
    table = get_db_table(device)

    conn = pymysql.connect(host=HOSTNAME, user=USERNAME, password=PASSWORD,
                           db=DEVICE_DB_NAME, charset='utf8')
    curs = conn.cursor()
    sql = "select dt," + track + " from " + table + " where " + track + " is not null and dt>%s and dt<%s and rosette=%s and bed = %s ;"
    curs.execute(sql, (time[0], time[-1], room[:-3], room[-2:]))
    row = curs.fetchall()
    conn.close()

    # print(len(time))
    # print(len(pcg_feature))

    row = np.array(row)
    if len(row) == 0:
        return [], [], []

    device_time = row[:, 0]
    dev_data = np.array(row[:, 1], dtype=np.float16)

    result_pcg_data = []

    result_data, result_time = [], []
    total_cnt = 0
    for i, timeset in enumerate(device_time):
        tmp_data = []

        for cnt in range(total_cnt, len(time)):

            if timeset > time[cnt] - datetime.timedelta(minutes=5):
                if timeset - datetime.timedelta(minutes=15) < time[cnt] - datetime.timedelta(minutes=5):
                    tmp_data.append(pcg_feature[cnt])
                else:
                    total_cnt = cnt



            else:
                if not tmp_data:
                    break
                result_time.append(timeset)
                result_data.append(dev_data[i])
                # print(tmp_data)
                result_pcg_data.append(np.median(tmp_data))

                break

    return result_time, result_pcg_data, result_data



def draw_multi(results, parameter_names, compare_parameter, results_corr, path, filter_size=17,append_title=''):

    if parameter_names[0] in results:
        a=1
        #print(parameter_names[0])
    else:
        print(parameter_names[0], ' have no data')
        return



    fig = plt.figure(figsize=(20, 10))
    ax2 = fig.subplots()
    for i in range(len(parameter_names) - 1):
        tmp_pcg = scipy.signal.savgol_filter(results[parameter_names[i]][1], filter_size, 0)
        ax2.plot(results[parameter_names[i]][0], tmp_pcg,
                 label=parameter_names[i] + '=' + str(results_corr[parameter_names[i]][0]))

    plt.legend(loc='upper left', fontsize='xx-large')
    ax2 = ax2.twinx()

    tmp_comp = scipy.signal.savgol_filter(compare_parameter[parameter_names[-1]][1], filter_size, 0)
    ax2.plot(compare_parameter[parameter_names[-1]][0], tmp_comp, 'r', marker='.', label=parameter_names[-1], alpha=0.4)

    if 'SVR' in compare_parameter:
        for i in range(len(compare_parameter['SVR'][1])):
            if compare_parameter['SVR'][1][i] >= 800:
                plt.axvline(x=compare_parameter['SVR'][0][i],color='yellowgreen',ls='--',linewidth = 1,alpha=0.2)
            else:
                plt.axvline(x=compare_parameter['SVR'][0][i],color='deepskyblue',ls='--',linewidth = 1,alpha=0.2)

    plt.legend(loc='upper right', fontsize='xx-large')
    titlepath = (path + parameter_names[0] + append_title).split('/')[-1]
    plt.title(titlepath)
    plt.savefig(path + parameter_names[0],bbox_inches='tight',pad_inches=0.2)
    plt.close()


def draw_multi_v2(results, parameter_names, compare_parameter, results_corr, path, filter_size=17,append_title=''):

    if parameter_names[0] in results:
        a=1
        #print(parameter_names[0])
    else:
        print(parameter_names[0], ' have no data')
        return


    fig = plt.figure(figsize=(20, 10))
    ax2 = fig.subplots()
    for i in range(len(parameter_names) - 1):
        tmp_pcg = scipy.signal.savgol_filter(results[parameter_names[i]][1], filter_size, 0)
        ax2.plot(results[parameter_names[i]][0], tmp_pcg,
                 label=parameter_names[i] + '=' + str(results_corr[parameter_names[i]][0]))

    plt.legend(loc='upper left', fontsize='xx-large')
    ax2 = ax2.twinx()

    tmp_comp = scipy.signal.savgol_filter(compare_parameter[parameter_names[-1]][1], filter_size, 0)
    ax2.plot(compare_parameter[parameter_names[-1]][0], tmp_comp, 'r', marker='.', label=parameter_names[-1], alpha=0.4)

    if 'SVR' in compare_parameter:
        for i in range(len(compare_parameter['SVR'][1])):
            if compare_parameter['SVR'][1][i] >= 800:
                plt.axvline(x=compare_parameter['SVR'][0][i],color='yellowgreen',ls='--',linewidth = 1,alpha=0.2)
            else:
                plt.axvline(x=compare_parameter['SVR'][0][i],color='deepskyblue',ls='--',linewidth = 1,alpha=0.2)



    plt.legend(loc='upper right', fontsize='xx-large')
    titlepath = (path + parameter_names[0] + append_title).split('/')[-1]
    plt.title(titlepath)
    plt.savefig(path + parameter_names[0],bbox_inches='tight',pad_inches=0.2)
    plt.close()


def draw_sample_pcg(room, date, savepath):
    time, pcg_file = load_pcg_data(room, date)
    sample_index = len(pcg_file) // 10

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(room + '_' + str(date), fontsize=25)
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        pcg_data = np.load(pcg_file[i * sample_index])
        pcg_data = np.array(pcg_data['arr_0'][0, :, 3][:3000])

        plt.xlabel(str(time[i * sample_index].time()))
        plt.xticks([])
        plt.plot(pcg_data)

    plt.savefig(savepath + 'Data.png',bbox_inches='tight',pad_inches=0.2)
    # plt.show()

def draw_sample_pcg_v2(room, date, savepath):
    time, pcg_file = load_pcg_data_v2(room, date)
    sample_index = len(pcg_file) // 10

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(room + '_' + str(date), fontsize=25)
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        pcg_data = np.load(pcg_file[i * sample_index])
        pcg_data = np.array(pcg_data['arr_0'][0, :, 3][:3000])

        plt.xlabel(str(time[i * sample_index].time()))
        plt.xticks([])
        plt.plot(pcg_data)

    plt.savefig(savepath + 'Data.png',bbox_inches='tight',pad_inches=0.2)
    # plt.show()

def draw_10seg(results, parameter_names, compare_parameter, results_corr, path, filter_size=17):

    if parameter_names[0] in results:
        a=1
        #print(parameter_names[0])
    else:
        print(parameter_names[0], ' have no data')
        return
    titlepath = (path + parameter_names[0]).split('/')[-1]
    sample_index = len(results[parameter_names[0]][0]) // 10
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(titlepath, fontsize=25)
    for j in range(10):
        ax2 = plt.subplot2grid((5, 2), (j // 2, j % 2), colspan=1)

        comp_time = compare_parameter[parameter_names[-1]][0][j * sample_index: (j + 1) * sample_index]
        comp_data = compare_parameter[parameter_names[-1]][1][j * sample_index: (j + 1) * sample_index]

        for i in range(len(parameter_names) - 1):
            time = results[parameter_names[i]][0][j * sample_index: (j + 1) * sample_index]
            pcg_data = results[parameter_names[i]][1][j * sample_index: (j + 1) * sample_index]
            tmp_pcg = scipy.signal.savgol_filter(pcg_data, filter_size, 0)
            tmp_comp = scipy.signal.savgol_filter(comp_data, filter_size, 0)

            cor = np.corrcoef([tmp_pcg, tmp_comp])
            stcor = str(round(cor[0, 1], 2))
            ax2.plot(time, tmp_pcg,
                     label=parameter_names[i] + '=' + stcor)

            ax2_1 = ax2.twinx()
            ax2_1.plot(comp_time, tmp_comp, 'r', marker='.', label=parameter_names[-1], alpha=0.4)
            ax2.legend(loc='upper left')
            ax2_1.legend(loc='upper right')

    # plt.title(titlepath)
    plt.savefig(path + parameter_names[0] + '_10seg.png',bbox_inches='tight',pad_inches=0.2)
    plt.close()


def exist_dic(dic, name):
    if dic.get(name):
        return dic[name]
    else:
        return ['']
