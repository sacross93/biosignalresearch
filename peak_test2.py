import jyLibrary as jy
import vitaldb as vrdb
import vr_reader_fix as vr
from scipy.signal import butter, lfilter, hilbert, chirp ,find_peaks
import matplotlib.pyplot as plt

room_file=jy.searchDateRoom('D-03',21,3,8)

vrfile=vr.VitalFile(room_file[0])

AUDIOtime,AUDIOdata = vrfile.get_samples('AUDIO')
hearttime,heartdata = vrfile.get_samples('AUDIO_HEART')
s1time,s1data = vrfile.get_numbers('S1')
s2time,s2data = vrfile.get_numbers('S2')
s12time,s12data = vrfile.get_numbers('S12_INTERVAL')
len(hearttime)

jy.plot(hearttime[len(hearttime)//2:len(hearttime)//2+10000],heartdata[len(hearttime)//2:len(hearttime)//2+10000])

plt.plot(s1data, heartdata[s1data], "xr")
plt.plot(heartdata)
plt.legend(['prominence'])
plt.show()

s1stamp=jy.timeChange(s1time,'timestamp',9)

plt.figure(figsize=(20, 10))
plt.plot(hearttime[len(hearttime)//2:len(hearttime)//2+10000],heartdata[len(hearttime)//2:len(hearttime)//2+10000])
plt.plot(s1stamp[len(s1stamp)//2:,s1data//2,"xr")
plt.show()



for i in range(len(s1stamp)) :
    if hearttime[2745612] == s1stamp[i] :
        print(i)
        break

plt.figure(figsize=(20, 10))
peaks, _ = find_peaks(heartdata[len(heartdata)//2:len(heartdata)//2+10000], distance=140, height=100)
plt.plot(heartdata[len(heartdata)//2:len(heartdata)//2+10000])
plt.plot(peaks, heartdata[peaks+len(heartdata)//2], "xr")
plt.legend(['prominence'])
plt.show()

