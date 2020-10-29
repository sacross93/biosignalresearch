import numpy as np


def save_signal_data(path, data,status="wb"):
    data = np.array(data, dtype=np.float32).tobytes()
    with open(path, status)as f:
        f.write((data))


def read_signal_data(path):
    with open(path,'rb')as f:
        content = f.read()
        x = np.array(np.frombuffer(content, dtype=np.float32))
    return x
