import jyLibrary as jy
import vr_reader_fix as vr
import matplotlib.pyplot as plt

room_name="D-01"
a = jy.searchDateRoom(room_name,21,1,18)

print(a[3])

aa=vr.VitalFile(a[3])

s1time,s1data = aa.get_numbers("S1")

plt.figure(figsize=(20, 10))
plt.plot(s1time[5000+5000:5500+5000],s1data[5000+5000:5500+5000])
plt.show()

