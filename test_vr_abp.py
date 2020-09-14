import vr_reader_fix as vr
import check_filepath as cf
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2

default_path = "/mnt/Data/CloudStation/"
file = 'D-02/200221/'

#os.listdir(default_path+file)

selcted_file = default_path + file + 'D-02_200221_082405.vital'
vrfile = vr.VitalFile(selcted_file)

#time, ABP = vrfile.get_samples('PLETH')

#plt.plot(time[1000000:1000000+1250],ABP[1000000:1000000+1250])
#plt.show()

#time, temp = vrfile.get_numbers('AWAY_RR')
#plt.plot(time,temp)
#plt.show()
rooms = ['D-02']
date = ['200221']
a = cf.search_filepath(rooms,date)
aa=a.pop()
vrfileTemp=vr.VitalFile(aa)
time, temp = vrfileTemp.get_numbers('SV','EV1000')
#time2, temp2 = vrfileTemp.get_numbers('SV','Vigilance')
print(time)
print(temp)
plt.plot(time,temp)
plt.show()
#plt2.plot(time2,temp2)
#plt2.show()

#data =vrfileTemp.find_track('SVV')

