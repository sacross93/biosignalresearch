from scipy.signal import butter, lfilter
from scipy import signal





#low 10 high 100 fs=data(len)

def butter_bandpass(lowcut, highcut, fs, order=5) :
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b,a

def butter_bandpass_filter(data,lowcut,highcut,fs,order=5) :
    b, a = butter_bandpass(lowcut,highcut,fs,order=order)
    y = lfilter(b,a,data)
    return y

def notch_pass_filter(data,center,interval=20,sr=44100,normalized=False) :
    center = center/(sr/2) if normalized else center
    b,a = signal.irrnotch(center,center/interval,sr)
    filtered_data = signal.lfilter(b,a,data)
    return filtered_data


