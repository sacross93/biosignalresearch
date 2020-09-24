import numpy as np
import itertools

a=['/mnt/CloudStation/D-01/170804/D-01_170804_222307.vital', '/mnt/CloudStation/D-01/170804/D-01_170804_090336.vital', '/mnt/CloudStation/D-01/170804/D-01_170804_074044.vital', '/mnt/CloudStation/D-01/170804/D-01_170804_230433.vital'],['/mnt/CloudStation/D-01/170823/D-01_170823_094400.vital', '/mnt/CloudStation/D-01/170823/D-01_170823_084010.vital', '/mnt/CloudStation/D-01/170823/D-01_170823_073616.vital']


a = np.array(a).flatten().tolist()
a= list(itertools.chain(*a))

print(a[0])
