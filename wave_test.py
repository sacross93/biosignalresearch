import jyLibrary as jy
import vitaldb as vital
import vr_reader_fix as vr

d_day=jy.searchDateRoom('D-06',21,4,22)


trk_list=vital.vital_trks(d_day[0])

for i in trk_list :
    print(i)

vr_file=vr.VitalFile(d_day[0])

ECG_time,ECG_data=vr_file.get_samples('IBP5')
