import jyLibrary_fix as jy

room_name="F-08"
room_year=20
room_month=8
room_day=25
machine_name="None"
data_name="AUDIO"


room_date = jy.searchDateRoom(room_name,room_year,room_month,room_day)
atime, adata = jy.findMachineInfo(room_date, machine_name, data_name)


