import re
import datetime
from os import listdir, rename
from os.path import isdir, isfile, join, basename

beds_migration = (
    'B-01', 'B-02', 'B-03', 'B-04', 'C-01', 'C-02', 'C-03', 'C-04', 'C-05', 'C-06',
    'D-01', 'D-02', 'D-03', 'D-04', 'D-05', 'D-06', 'Y-01', 'OB-01', 'OB-02',
    'E-01', 'E-02', 'E-03', 'E-04', 'E-05', 'E-06', 'E-07', 'E-08', 'E-09', 'E-10',
    'F-01', 'F-02', 'F-03', 'F-04', 'F-05', 'F-06', 'F-07', 'F-08', 'F-09', 'F-10',
    'G-01', 'G-02', 'G-03', 'G-04', 'G-05', 'G-06',
    'H-01', 'H-02', 'H-03', 'H-04', 'H-05', 'H-06', 'H-07', 'H-08', 'H-09',
    'I-01', 'I-02', 'I-03', 'I-04', 'J-01', 'J-02', 'J-03', 'J-04', 'J-05', 'J-06',
    'K-01', 'K-02', 'K-03', 'K-04', 'K-05', 'K-06', 'IPACU-01', 'IPACU-02',
    'PICU1-01', 'PICU1-02', 'PICU1-03', 'PICU1-04', 'PICU1-05', 'PICU1-06',
    'PICU1-07', 'PICU1-08', 'PICU1-09', 'PICU1-10', 'PICU1-11',
    'WREC-01', 'WREC-02', 'WREC-03', 'WREC-04', 'WREC-05', 'WREC-06', 'WREC-07', 'WREC-08', 'WREC-09', 'WREC-10',
    'WREC-11', 'WREC-12', 'WREC-13', 'WREC-14', 'WREC-15', 'EREC-01', 'EREC-02', 'EREC-03', 'EREC-04', 'EREC-05',
    'EREC-06', 'EREC-07', 'EREC-08', 'EREC-09', 'EREC-10', 'EREC-11', 'EREC-12', 'EREC-13', 'EREC-14', 'EREC-15',
    'EREC-16', 'EREC-17', 'EREC-18', 'EREC-19', 'NREC-01', 'NREC-02', 'NREC-03', 'NREC-04', 'NREC-05', 'NREC-06',
    'NREC-07', 'NREC-08', 'NREC-09', 'NREC-10', 'NREC-11', 'NREC-12', 'NREC-13', 'NREC-14', 'NREC-15', 'NREC-16'
)

data_root = '/volume3/Backup/CloudStation'

dt_re = re.compile('[0-9]{6}')
file_re = re.compile('[0-9]{6}_[0-9]{6}.vital')

for bed_name in beds_migration:
    if isdir(join(data_root, bed_name)):
        for dt in listdir(join(data_root, bed_name)):
            if dt_re.match(dt) and isdir(join(data_root, bed_name, dt)):
                for file in listdir(join(data_root, bed_name, dt)):
                    if isfile(join(data_root, bed_name, dt, file)):
                        if file_re.search(file) and not file.startswith(bed_name+'_'+dt+'_'):
#                            rename(join(data_root, bed_name, dt, file), join(data_root, bed_name, dt, bed_name + '_' + file[-19:]))
                            print(bed_name, dt, file, bed_name+'_'+file[-19:])

