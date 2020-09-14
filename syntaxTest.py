import multiprocessing
import parmap
import numpy as np
import vr_reader_fix as vr
import check_filepath as cf
import os
import matplotlib.pyplot as plt
import datetime
import copy
import time

a=datetime.datetime(2020,8,25,10,20,00)
print(a)
b=datetime.datetime.timestamp(a)
print(b)