import numpy
import vr_reader_fix as vr
import os
import jyLibrary_fix as jy
from multiprocessing import *
import multiprocessing

a = jy.searchDateRoom("D-01",21,4)

num_cores = multiprocessing.cpu_count()

print(num_cores)

with Pool(num_cores//2) as p :
    multi_result = p.map(vr.VitalFile,a)

