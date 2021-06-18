import os



path = os.getcwd()
print(path)
for filename in os.listdir(path+"/python_program/test_rename"):
    os.rename(path+"/python_program/test_rename/"+filename,path+"/python_program/test_rename/"+"F-01_"+filename[-19:])


