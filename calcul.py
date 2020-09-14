import numpy as np
import pandas as pd


a=np.arange(1,21)

print(a)
print(len(a))
Q1=len(a)*0.25
print(Q1)
print(a[int(Q1)-1])
Q2 = len(a)*0.5
print(Q2)
print(a[int(Q2)])
print("")

b=np.arange(1,27)
print(b)
Q1=len(b)*0.25
print(Q1)
if Q1 % 1 != 0 :
    print(int(Q1))
    print(int(Q1)+1)
    print("")
    print(b[int(Q1)])
    print(b[int(Q1)+1])
    Q1 = (b[int(Q1)]+b[int(Q1)+1])/2
    print(Q1)

Q2 = np.quantile(b,0.5) #? 13.5...?
print(Q2)
Q3 = np.quantile(b,0.75) #? 19.75 .. ?
print(Q3)
