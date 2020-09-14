import os
#import pandas as pd
import numpy as np
from scipy import signal

import matplotlib.pyplot as plt
import math
import scipy.stats as stat
from scipy.signal import hilbert
#import peakutils
import time
import datetime
import gzip
import pickle
#import arr
#import pywt
import check_filepath
import scipy


def bandpass_filter(signal,HZ,rf,uf):

    gap = len(signal)//HZ
    fftdata = np.fft.fft(signal,len(signal))
    fftdata[0:rf*gap] = 0
    fftdata[uf*gap:-uf * gap] = 0
    fftdata[-rf*gap:]=0
    result =  np.fft.ifft(fftdata,len(signal))
    print(len(result))
    return result


def cal_variance(datalist):
    if len(datalist[0]) == 0:
        return np.float64('nan')
    var = []
    for data in datalist:
        max = np.max(data)
        min = np.min(data)
        var.append((max - min) / np.mean([max, min]))
    return float(np.mean(var))


# segmentation 10 and check max
def check_data(pcg):
    hz = len(pcg) // 5
    pcg_max = 0
    for i in range(5):
        pcg_local_max = np.max(pcg[hz * i:hz * (i + 1)])

        if pcg_max > pcg_local_max:
            pcg_max = pcg_local_max

        else:
            if pcg_max > pcg_local_max*2:
                return 1
        # if pcg_max < pcg_local_max:
        #    pcg_max = pcg_local_max
        #print(pcg_local_max)
        if pcg_local_max < 0.5:
            return 1
    if max(pcg) < 1:
        return 1

    return 0


# segmentation 5 and check max
def check_data_10(pcg):
    hz = len(pcg) // 5
    pcg_max = 0
    for i in range(5):
        pcg_local_max = np.max(pcg[hz * i:hz * (i + 1)])
        # if pcg_max < pcg_local_max:
        #    pcg_max = pcg_local_max

        if pcg_local_max < 0.5:
            return 1

    return 0


# �@ �� Hh
def median_5min(time, data, window=5):
    result_data = []

    for i in range(window, len(time)):

        cnt = i
        while (1):
            if cnt == -1:
                break
            if datetime.datetime.combine(datetime.date.today(), time[i]) - datetime.timedelta(
                    minutes=window) <= datetime.datetime.combine(datetime.date.today(), time[cnt]):
                result_data.append(data[cnt])
                cnt = cnt - 1
                continue

            break

    return result_data


#
def median_5min_list(time, datalist, window=5):
    result_data = [[]]
    tmp_data = [[]]
    for i in range(1, len(datalist)):
        result_data.append([])
        tmp_data.append([])

    tmp = tmp_data

    for i in range(window, len(time)):
        tmp_data = tmp
        cnt = i
        while (1):
            if cnt == -1:
                break
            if datetime.datetime.combine(datetime.date.today(), time[i]) - datetime.timedelta(
                    minutes=window) <= datetime.datetime.combine(datetime.date.today(), time[cnt]):

                for j in range(len(result_data)):
                    tmp_data[j].append(datalist[j][cnt])
                cnt = cnt - 1
                continue

            break

        for j in range(len(result_data)):
            result_data[j].append(np.median(tmp_data[j]))

    return result_data


# only float data
def save_signal_data(path, data, status="wb"):
    data = np.array(data, dtype=np.float32).tobytes()
    with open(path, status)as f:
        f.write((data))


def read_signal_data(path):
    with open(path, 'rb')as f:
        content = f.read()
        x = np.array(np.frombuffer(content, dtype=np.float32))
    return x


def total_graph_corr_std(TARGET_PATH, FILE_DATE, DATE_TIME, S1_time, d1data, d2data, d3data, d4data, d1name, d2name,
                         d3name, d4name, state):
    d1 = d1data[:]
    d2 = d2data[:]
    d3 = d3data[:]
    d4 = d4data[:]

    d1_std = np.std(d1)

    d2_std = np.std(d2)
    d3_std = np.std(d3)
    d4_std = np.std(d4)

    d1_mean = np.mean(d1)
    d2_mean = np.mean(d2)
    d3_mean = np.mean(d3)
    d4_mean = np.mean(d4)

    print(len(d3))

    d1_2 = d1[:]
    d1_3 = d1[:]
    d1_4 = d1[:]
    S1_time_2 = S1_time[:]
    S1_time_3 = S1_time[:]
    S1_time_4 = S1_time[:]

    z = 0
    while (z < len(d2)):

        if len(d2) == z:
            break
        std1 = np.abs(d1_2[z] - d1_mean)
        std2 = np.abs(d2[z] - d2_mean)

        if std1 > d1_std * 2.53 or std1 < d1_std / 2.53 or std2 > d2_std * 2.53 or std2 < d2_std / 2.53:
            d2.pop(z)
            d1_2.pop(z)
            S1_time_2.pop(z)
            continue
        z = z + 1

    z = 0
    while (z < len(d3)):

        if len(d3) == z:
            break
        std1 = np.abs(d1_3[z] - d1_mean)
        std3 = np.abs(d3[z] - d3_mean)

        if std1 > d1_std * 2.5 or std1 < d1_std / 2.5 or std3 > d3_std * 2.5 or std3 < d3_std / 2.5:
            d3.pop(z)
            d1_3.pop(z)
            S1_time_3.pop(z)
            continue

        z = z + 1

    print(len(d3))
    print(len(d1_3))
    len(S1_time_3)

    z = 0
    while (z < len(d4)):

        if len(d4) == z:
            break
        std1 = np.abs(d1_4[z] - d1_mean)
        std4 = np.abs(d4[z] - d4_mean)

        if std1 > d1_std * 2.53 or std1 < d1_std / 2.53 or std4 > d4_std * 2.53 or std4 < d4_std / 2.53:
            d1_4.pop(z)
            d4.pop(z)
            S1_time_4.pop(z)

        z = z + 1

    fig = plt.figure(figsize=(20, 10))
    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=4)
    ax2 = plt.subplot2grid((4, 4), (1, 0), colspan=4)
    ax3 = plt.subplot2grid((4, 4), (2, 0), colspan=4)
    ax4 = plt.subplot2grid((4, 4), (3, 0), colspan=2, rowspan=2)
    ax5 = plt.subplot2grid((4, 4), (3, 2), colspan=2, rowspan=2)

    corr_S12 = np.corrcoef(d1_2, d2)[0, 1]
    corr_cS12 = np.corrcoef(d1_3, d3)[0, 1]
    corr_S12w = np.corrcoef(d1_4, d4)[0, 1]

    S1_time = S1_time_2
    d1 = d1_2
    ax1.set_title("total_corr_" + d1name + "_" + d2name + " :" + str(corr_S12)[:5])
    ax1.plot(S1_time, d2, label=d2name, marker='s')
    ax1_2 = ax1.twinx()
    ax1_2.plot(S1_time, d1, label=d1name, marker='s', color='r')
    ax1.legend(loc='upper right')
    ax1_2.legend(loc='upper left')

    S1_time = S1_time_3
    d1 = d1_3
    ax2.set_title(d1name + "_" + d3name + " :" + str(corr_cS12)[:5])
    ax2.plot(S1_time, d3, label=d3name, marker='s')
    ax2_2 = ax2.twinx()
    ax2_2.plot(S1_time, d1, label=d1name, marker='s', color='r')
    ax2.legend(loc='upper right')
    ax2_2.legend(loc='upper left')

    S1_time = S1_time_4
    d1 = d1_4
    ax3.set_title(d1name + "_" + d4name + " :" + str(corr_S12w)[:5])
    ax3.plot(S1_time, d4, label=d4name, marker='s')

    ax3_2 = ax3.twinx()
    ax3_2.plot(S1_time, d1, label=d1name, marker='s', color='r')
    ax3.legend(loc='upper right')
    ax3_2.legend(loc='upper left')

    ax4.hist(d1, label=d1name, bins=40)
    ax4.legend(loc='upper left')

    ax5.hist(d2, label=d2name, bins=40)
    ax5.legend(loc='upper left')

    plt.savefig(
        TARGET_PATH + "/" + FILE_DATE + "/" + "total_result/" + FILE_DATE + "_" + DATE_TIME + "_total_corr_" + d1name + "_" + d2name + state + ".png",
        dpi=200)
    plt.close()


def total_graph_corr_4(TARGET_PATH, FILE_DATE, DATE_TIME, S1_time, d1, d2, d3, d4, d5, d1name, d2name, d3name, d4name,
                       d5name, state):
    # S12_PP
    fig = plt.figure(figsize=(20, 15))
    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=4)
    ax2 = plt.subplot2grid((4, 4), (1, 0), colspan=4)
    ax3 = plt.subplot2grid((4, 4), (2, 0), colspan=4)
    ax4 = plt.subplot2grid((4, 4), (3, 0), colspan=4)

    corr_S12 = np.corrcoef(d1, d2)[0, 1]
    corr_cS12 = np.corrcoef(d1, d3)[0, 1]
    corr_S12w = np.corrcoef(d1, d4)[0, 1]
    corr_d5 = np.corrcoef(d1, d5)[0, 1]

    ax1.set_title(state + "_" + d1name + "_" + d2name + " :" + str(corr_S12)[:5])
    ax1.plot(S1_time, d2, label=d2name, marker='s')
    ax1_2 = ax1.twinx()
    ax1_2.plot(S1_time, d1, label=d1name, marker='s', color='r')
    ax1.legend(loc='upper left')
    ax1_2.legend(loc='upper right')
    ax1.set(xlabel='')

    ax2.set_title(d1name + "_" + d3name + " :" + str(corr_cS12)[:5])
    ax2.plot(S1_time, d3, label=d3name, marker='s')
    ax2_2 = ax2.twinx()
    ax2_2.plot(S1_time, d1, label=d1name, marker='s', color='r')
    ax2.legend(loc='upper left')
    ax2_2.legend(loc='upper right')
    ax1.set(xlabel='')

    ax3.set_title(d1name + "_" + d4name + " :" + str(corr_S12w)[:5])
    ax3.plot(S1_time, d4, label=d4name, marker='s')
    ax3_2 = ax3.twinx()
    ax3_2.plot(S1_time, d1, label=d1name, marker='s', color='r')
    ax3.legend(loc='upper left')
    ax3_2.legend(loc='upper right')
    ax1.set(xlabel='')

    ax4.set_title(d1name + "_" + d5name + " :" + str(corr_d5)[:5])
    ax4.plot(S1_time, d5, label=d5name, marker='s')
    ax4_2 = ax4.twinx()
    ax4_2.plot(S1_time, d1, label=d1name, marker='s', color='r')
    ax4.legend(loc='upper left')
    ax4_2.legend(loc='upper right')

    plt.savefig(
        TARGET_PATH + "/" + FILE_DATE + "/" + "total_result/" + FILE_DATE + "_" + DATE_TIME + "_total_corr_" + d1name + "_" + d2name + state + ".png",
        dpi=200)
    plt.close()


def total_graph_corr_2(TARGET_PATH, FILE_DATE, DATE_TIME, S1_time, d1, d2, d3, d1name, d2name, d3name, state):
    # S12_PP
    fig = plt.figure(figsize=(20, 10))
    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=4)
    ax2 = plt.subplot2grid((2, 4), (1, 0), colspan=4)

    corr_S12 = np.corrcoef(d1, d2)[0, 1]
    corr_cS12 = np.corrcoef(d1, d3)[0, 1]

    ax1.set_title(state + "_" + d1name + "_" + d2name + " :" + str(corr_S12)[:5])
    ax1.plot(S1_time, d2, label=d2name, marker='s')
    ax1_2 = ax1.twinx()
    ax1_2.plot(S1_time, d1, label=d1name, marker='s', color='r')
    ax1.legend(loc='upper left')
    ax1_2.legend(loc='upper right')
    ax1.set(xlabel='')

    ax2.set_title(d1name + "_" + d3name + " :" + str(corr_cS12)[:5])
    ax2.plot(S1_time, d3, label=d3name, marker='s')
    ax2_2 = ax2.twinx()
    ax2_2.plot(S1_time, d1, label=d1name, marker='s', color='r')
    ax2.legend(loc='upper left')
    ax2_2.legend(loc='upper right')
    ax1.set(xlabel='')

    plt.savefig(
        TARGET_PATH + "/" + FILE_DATE + "/" + "total_result/" + FILE_DATE + "_" + DATE_TIME + "_total_corr_" + d1name + "_" + d2name + state + ".png",
        dpi=200)
    plt.close()


def total_graph_corr(TARGET_PATH, FILE_DATE, DATE_TIME, S1_time, d1, d2, d3, d4, d1name, d2name, d3name, d4name, state):
    # S12_PP
    fig = plt.figure(figsize=(20, 10))
    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=4)
    ax2 = plt.subplot2grid((4, 4), (1, 0), colspan=4)
    ax3 = plt.subplot2grid((4, 4), (2, 0), colspan=4)
    ax4 = plt.subplot2grid((4, 4), (3, 0), colspan=2, rowspan=2)
    ax5 = plt.subplot2grid((4, 4), (3, 2), colspan=2, rowspan=2)

    corr_S12 = np.corrcoef(d1, d2)[0, 1]
    corr_cS12 = np.corrcoef(d1, d3)[0, 1]
    corr_S12w = np.corrcoef(d1, d4)[0, 1]

    ax1.set_title(state + "_" + d1name + "_" + d2name + " :" + str(corr_S12)[:5])
    ax1.plot(S1_time, d2, label=d2name, marker='s')
    ax1_2 = ax1.twinx()
    ax1_2.plot(S1_time, d1, label=d1name, marker='s', color='r')
    ax1.legend(loc='upper right')
    ax1_2.legend(loc='upper left')

    ax2.set_title(d1name + "_" + d3name + " :" + str(corr_cS12)[:5])
    ax2.plot(S1_time, d3, label=d3name, marker='s')
    ax2_2 = ax2.twinx()
    ax2_2.plot(S1_time, d1, label=d1name, marker='s', color='r')
    ax2.legend(loc='upper right')
    ax2_2.legend(loc='upper left')

    ax3.set_title(d1name + "_" + d4name + " :" + str(corr_S12w)[:5])
    ax3.plot(S1_time, d4, label=d4name, marker='s')

    ax3_2 = ax3.twinx()
    ax3_2.plot(S1_time, d1, label=d1name, marker='s', color='r')
    ax3.legend(loc='upper right')
    ax3_2.legend(loc='upper left')

    ax4.hist(d1, label=d1name, bins=40, range=(min(d1), max(d1)))
    ax4.legend(loc='upper left')

    ax5.hist(d2, label=d2name, bins=40, range=(min(d2), max(d2)))
    ax5.legend(loc='upper left')

    plt.savefig(
        TARGET_PATH + "/" + FILE_DATE + "/" + "total_result/" + FILE_DATE + "_" + DATE_TIME + "_total_corr_" + d1name + "_" + d2name + state + ".png",
        dpi=200)
    plt.close()


def total_graph_corr_sv(TARGET_PATH, FILE_DATE, DATE_TIME, S1_time, S1_time2, d1, d2, d3, d4, d1name, d2name, d3name,
                        d4name, state):
    # S12_PP
    fig = plt.figure(figsize=(20, 10))
    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=4)
    ax2 = plt.subplot2grid((4, 4), (1, 0), colspan=4)
    ax3 = plt.subplot2grid((4, 4), (2, 0), colspan=4)
    ax4 = plt.subplot2grid((4, 4), (3, 0), colspan=2, rowspan=2)
    ax5 = plt.subplot2grid((4, 4), (3, 2), colspan=2, rowspan=2)

    corr_S12 = np.corrcoef(d1, d2)[0, 1]
    corr_cS12 = np.corrcoef(d1, d3)[0, 1]
    corr_S12w = np.corrcoef(d1, d4)[0, 1]

    ax1.set_title("total_corr_" + d1name + "_" + d2name + " :" + str(corr_S12)[:5])
    ax1.plot(S1_time, d2, label=d2name, marker='s')
    ax1_2 = ax1.twinx()
    ax1_2.plot(S1_time, d1, label=d1name, marker='s', color='r')
    ax1.legend(loc='upper right')
    ax1_2.legend(loc='upper left')

    ax2.set_title(d1name + "_" + d3name + " :" + str(corr_cS12)[:5])
    ax2.plot(S1_time2, d3, label=d3name, marker='s')
    ax2_2 = ax2.twinx()
    ax2_2.plot(S1_time2, d1, label=S1_time2, marker='s', color='r')
    ax2.legend(loc='upper right')
    ax2_2.legend(loc='upper left')

    ax3.set_title(d1name + "_" + d4name + " :" + str(corr_S12w)[:5])
    ax3.plot(S1_time, d4, label=d4name, marker='s')

    ax3_2 = ax3.twinx()
    ax3_2.plot(S1_time, d1, label=d1name, marker='s', color='r')
    ax3.legend(loc='upper right')
    ax3_2.legend(loc='upper left')

    ax4.hist(d1, label=d1name, bins=40, range=(min(d1), max(d1)))
    ax4.legend(loc='upper left')

    ax5.hist(d2, label=d2name, bins=40, range=(min(d2), max(d2)))
    ax5.legend(loc='upper left')

    plt.savefig(
        TARGET_PATH + "/" + FILE_DATE + "/" + "total_result/" + FILE_DATE + "_" + DATE_TIME + "_total_corr_" + d1name + "_" + d2name + state + ".png",
        dpi=200)
    plt.close()


def bland_altman_plot(data1, data2, *args, **kwargs):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')


def median_mean(data):
    tmp_median = data[:]
    tmp_median.sort()
    result = (np.mean(tmp_median[len(tmp_median) // 5:(len(tmp_median) * 4) // 5]))
    return result


def search_pcg_area(pcg, HZ, S1_index, S2_index):
    coef, freqs = pywt.cwt(pcg, 15, 'cgau2')

    pcg_filtered = arr.band_pass(coef[0], HZ, 1, HZ // 2)

    pcg100_hil = hilbert(pcg_filtered)
    pcg100_hilabs = np.abs(pcg100_hil)
    pcg_area = scipy.signal.savgol_filter(pcg100_hilabs, 55, 0)

    S1window_peak, S2window_peak, S12int_window = [], [], []
    S1_st, S1_et, S2_st, S2_et = [], [], [], []

    for i in range(len(S1_index)):

        S1 = S1_index[i]
        S2 = S2_index[i]

        breakflag = 0
        cnt = 0
        downflag = 0
        upflag = 0
        # search S1_peak
        # print('S1_peak')
        while (1):
            # print(pcg_area[S1 + cnt] , pcg_area[S1 + cnt + 1])
            # print(cnt,pcg_area[S1 + cnt],pcg_area[S2 + cnt + 1])

            if pcg_area[S1 + cnt] < pcg_area[S1 + cnt + 1]:
                # print(cnt, downflag)
                if downflag == 1:
                    S1window_index = S1 + cnt + 1
                    S1_peak = pcg_area[S1 + cnt + 1]
                    break
                cnt = cnt + 1
                upflag = 1
                continue

            if pcg_area[S1 + cnt] == pcg_area[S2 + cnt + 1]:
                # print(cnt, pcg_area[S1 + cnt], pcg_area[S2 + cnt + 1], 3)
                if breakflag == 1:
                    break
                if cnt == 0:
                    cnt = cnt + 5
                    breakflag = 1
                    continue
                elif cnt < 0:
                    cnt = cnt - 5
                    breakflag = 1
                    continue
                else:
                    cnt = cnt + 5
                    breakflag = 1
                    continue


            else:
                # print(cnt, upflag)
                if upflag == 1:
                    S1window_index = S1 + cnt
                    S1_peak = pcg_area[S1 + cnt]
                    break
                cnt = cnt - 1

                downflag = 1
                continue

            # print(cnt,pcg_area[S1 + cnt],pcg_area[S2 + cnt + 1],2)

        if breakflag == 1:
            continue
        breakflag = 0
        # print('S2_peak')
        cnt = 0
        downflag = 0
        upflag = 0
        while (1):

            if pcg_area[S2 + cnt] < pcg_area[S2 + cnt + 1]:
                if downflag == 1:
                    S2window_index = S2 + cnt + 1
                    S2_peak = pcg_area[S2 + cnt + 1]
                    break

                cnt = cnt + 1
                upflag = 1
                continue
            if pcg_area[S2 + cnt + 1] < pcg_area[S2 + cnt]:
                if upflag == 1:
                    S2window_index = S2 + cnt
                    S2_peak = pcg_area[S2 + cnt]
                    break
                cnt = cnt - 1
                downflag = 1
                continue

            if pcg_area[S2 + cnt] == pcg_area[S2 + cnt + 1]:
                if cnt == 0:
                    cnt = cnt + 5
                    continue
                elif cnt < 0:
                    cnt = cnt - 5
                else:
                    cnt = cnt + 5
        # search st, et

        if breakflag == 1:
            continue

        S12int_window.append(S2window_index - S1window_index)

        # print(S1_peak)
        # print('st1')
        stflag = 0
        j = 0
        tmpS1st, tmpS1et, tmps2et, tmps2st = 0, 0, 0, 0
        while (j < 300):
            # print(S1_peak,pcg_area[S1-j])
            if pcg_area[S1window_index - j] < S1_peak / 2:
                tmpS1st = (S1 - j)
                stflag = 1
                break
            j = j + 1
        # print('st2')
        j = 0
        while (j < 300 and stflag == 1):
            if pcg_area[S1window_index + j] < S1_peak / 2:
                tmpS1et = (S1window_index + j)
                break
            j = j + 1
        j = 0
        # print('st3')
        # print(S2_peak)
        stflag = 0
        while (j < 300):
            if pcg_area[S2window_index - j] < S2_peak / 2:
                tmps2st = (S2window_index - j)
                stflag = 1
                break
            j = j + 1
        j = 0
        # print('st4')
        while (j < 300 and stflag == 1):
            if pcg_area[S2window_index + j] < S2_peak / 2:
                tmps2et = (S2window_index + j)
                break
            j = j + 1

        if tmpS1st != 0 and tmps2st != 0:
            S1_st.append(tmpS1st)
            S1_et.append(tmpS1et)
            S2_st.append(tmps2st)
            S2_et.append(tmps2et)

        S1window_peak.append(S1_peak)
        S2window_peak.append(S2_peak)

    return pcg_area, S1window_peak, S2window_peak, S12int_window, S1_st, S1_et, S2_st, S2_et


def spectrofft(data, sec, HZ):
    # Before FFT :
    Ts = 1 / HZ  # Sampling interval
    Fs = 1 / Ts  # Sampling rate = 1000
    n = 1000
    k = np.arange(n)  # 0 ~ 1000
    # Set arrays
    freq = []
    X = []
    Y = []
    T = list(range(0, sec))  # T = {0, 1, 2, 3, ..., 30}
    # Declare X and Y axis
    for i in T:
        Y.append(data[i * HZ:(i + 1) * HZ])  # measuring point to 30000
        freq.append(np.fft.fft(Y[i]) / 1000)
        freq[i] = freq[i][range(0, 500)]  # set frequency range for viewing
        freq[i] = abs(freq[i])
    freq = np.asarray(freq)
    freq = freq.transpose()
    freq[0] = 0
    return freq


def number_search(dt, number, datetime_start, datetime_end):
    p_start = 0
    while dt[p_start] < datetime_start if p_start < len(dt) else False:
        p_start += 1
    p_end = p_start
    while dt[p_end] < datetime_end if p_end < len(dt) else False:
        p_end += 1
    if p_start < p_end:
        return dt[p_start:p_end], number[p_start:p_end]
    return [], []


def number_search_index(dt, number, datetime_start, datetime_end):
    p_start = 0
    while dt[p_start] < datetime_start if p_start < len(dt) else False:
        p_start += 1
    p_end = p_start
    while dt[p_end] < datetime_end if p_end < len(dt) else False:
        p_end += 1
    if p_start < p_end:
        return p_start, p_end
    return 0, 0


def number_search(dt, number, datetime_start, datetime_end):
    p_start = 0
    while dt[p_start] < datetime_start if p_start < len(dt) else False:
        p_start += 1
    p_end = p_start
    while dt[p_end] < datetime_end if p_end < len(dt) else False:
        p_end += 1
    if p_start < p_end:
        return dt[p_start:p_end], number[p_start:p_end]
    return [], []

def sample_search(dt, number,datetime_end):
    p_start = 0
    while dt[p_start] < datetime_end if p_start < len(dt)-1000 else False:
        p_start += 1000

    while dt[p_start] < datetime_end if p_start < len(dt) else False:
        p_start += 1

    p_end = p_start

    if p_start !=0 :
        return dt[p_start-10000:p_end], number[p_start-10000:p_end]
    return [], []

def sample_search_60(dt, number,datetime_end):
    p_start = 0
    while dt[p_start] < datetime_end if p_start < len(dt)-1000 else False:
        p_start += 1000

    while dt[p_start] < datetime_end if p_start < len(dt) else False:
        p_start += 1

    p_end = p_start

    if p_start !=0 :
        return dt[p_start-60000:p_end], number[p_start-60000:p_end]
    return [], []

def sample_search_second(dt, number,datetime_end,hz,second):
    p_start = 0
    while dt[p_start] < datetime_end if p_start < len(dt)-hz else False:
        p_start += hz

    while dt[p_start] < datetime_end if p_start < len(dt) else False:
        p_start += 1

    p_end = p_start

    if p_start !=0 :
        return dt[p_start-second*hz:p_end], number[p_start-second*hz:p_end]
    return [], []


def sample_search_SV(dt, number,datetime_end):
    p_start = 0
    while dt[p_start] < datetime_end if p_start < len(dt) else False:
        p_start += 1
    p_end = p_start

    if p_start !=0 :
        return dt[p_start-15000:p_end], number[p_start-15000:p_end]
    return [], []
'''
def wave_filter(abp, pcg, ecg):
    dif = np.diff(abp)
    error = []
    for i in range(3, len(dif) - 3):
        if dif[i] < -1:
            error.append(i + 1)
    if len(error) == 0:
        return abp, pcg, ecg
    if error[0] < 100:
        error.pop(0)
    if error[-1] > 49900:
        error.pop(-1)
    for err in error:
        pcg[err] = (pcg[err - 3] + pcg[err + 3]) / 2
        pcg[err + 1] = (pcg[err - 2] + pcg[err + 4]) / 2
        pcg[err + 2] = (pcg[err - 1] + pcg[err + 5]) / 2
        ecg[err] = (ecg[err - 3] + ecg[err + 3]) / 2
        ecg[err + 1] = (ecg[err - 2] + ecg[err + 4]) / 2
        ecg[err + 2] = (ecg[err - 1] + ecg[err + 5]) / 2
        abp[err] = (abp[err - 3] + abp[err + 3]) / 2
        abp[err + 1] = (abp[err - 2] + abp[err + 4]) / 2
        abp[err + 2] = (abp[err - 1] + abp[err + 5]) / 2
    return abp, pcg, ecg
'''


def wave_filter(abp, pcg, ecg):
    for i in range(1, len(abp)):
        if abp[i] < 1:
            abp[i] = abp[i - 1]
            pcg[i] = pcg[i - 1]
            ecg[i] = ecg[i - 1]

    return abp, pcg, ecg


def wave_filter2(abp, pcg):
    for i in range(1, len(abp)):
        if abp[i] < 1:
            abp[i] = abp[i - 1]
            pcg[i] = pcg[i - 1]

    return abp, pcg




def PCG_shannon_peaks(a_peaks, up_peaks, test_pcg, HZ):#PCG_detect_abp

    shannon = []
    # shannon.append(test_pcg[0])
    for i in range(0, len(test_pcg)):

        En = test_pcg[i - 10:i + 10] * test_pcg[i - 10:i + 10] * np.log10(
            test_pcg[i - 10:i + 10] * test_pcg[i - 10:i + 10])
        Ean = np.nansum(En) / 20
        mEa = np.mean(En)
        sEa = np.std(En)

        tmp_sha = (Ean - mEa) / sEa

        shannon.append(Ean)

    shannon = np.abs(shannon)


    sink = 0
    while up_peaks[sink] < a_peaks[0]:
        sink += 1
        if sink >= len(up_peaks):
            break
    dp = []
    up_peaks = up_peaks[sink:]
    for i in range(len(up_peaks)):
        for j in range(len(a_peaks)):
            pitv = up_peaks[i] - a_peaks[j]
            if pitv < 0:
                dp.append(a_peaks[j - 1])
                break
    a_peaks = dp
    s1_index = []
    s2_index = []


    for i in range(1, len(a_peaks) - 1):
        itv = up_peaks[i] - a_peaks[i]
        if a_peaks[i] - itv * 1.6 < 0:
            continue
        # aitv = int((a_peaks[i + 1] - a_peaks[i])// 3)
        S1 = a_peaks[i] - int(itv * 1.6) + np.argmax(shannon[a_peaks[i] - int(itv * 1.6):a_peaks[i]])
        S2 = up_peaks[i] - int(itv * 0.05) + np.argmax(shannon[up_peaks[i] - int(itv * 0.05):up_peaks[i] + 150])
        s1_index.append(S1)
        s2_index.append(S2)


    s1_st,s1_et = [],[]
    s2_st,s2_et = [],[]
    Shannon_S1_Area,Shannon_S2_Area = [],[]

    #search Area
    for i in range(len(s1_index)):
        s1_peak = shannon[s1_index[i]]
        s2_peak = shannon[s2_index[i]]

        for j in range(1,100):
            if shannon[s1_index[i] -j] <s1_peak/4:
                tmp_st1_index = s1_index[i]-j
                break
            tmp_st1_index = np.nan
        s1_st.append(tmp_st1_index)

        for j in range(1,100):
            if shannon[s2_index[i] -j] <s2_peak/3:
                tmp_st2_index = s2_index[i]-j
                break
            tmp_st2_index = np.nan
        s2_st.append(tmp_st2_index)

        for j in range(1,100):
            if shannon[s1_index[i] +j] <s1_peak/4:
                tmp_et1_index = s1_index[i]+j
                break
            tmp_et1_index = np.nan
        s1_et.append(tmp_et1_index)

        for j in range(1,100):
            if shannon[s2_index[i] +j] <s2_peak/3:
                tmp_et2_index = s2_index[i]+j
                break
            tmp_et2_index = np.nan
        s2_et.append(tmp_et2_index)

        if tmp_st1_index is not np.nan and tmp_et1_index is not np.nan :
            Shannon_S1_Area.append(np.sum(shannon[tmp_st1_index:tmp_et1_index]))
        else :Shannon_S1_Area.append(np.nan)
        if  tmp_et2_index is not np.nan  and tmp_st2_index is not np.nan:
            #print(tmp_st2_index,tmp_et2_index)
            Shannon_S2_Area.append(np.sum(shannon[tmp_st2_index:tmp_et2_index]))
        else :Shannon_S2_Area.append(np.nan)

    return np.array(s1_st),np.array(s1_et),np.array(s2_st),np.array(s2_et),np.array(Shannon_S1_Area),np.array(Shannon_S2_Area)


def PCG_detect_abp(a_peaks, up_peaks, pcg, HZ):
    sink = 0
    while up_peaks[sink] < a_peaks[0]:
        sink += 1
        if sink >= len(up_peaks):
            break
    dp = []
    up_peaks = up_peaks[sink:]
    for i in range(len(up_peaks)):
        for j in range(len(a_peaks)):
            pitv = up_peaks[i] - a_peaks[j]
            if pitv < 0:
                dp.append(a_peaks[j - 1])
                break
    a_peaks = dp
    Psys = []
    Pdias = []
    pp = []
    s1_index = []
    s2_index = []
    s12i = []
    itv2 = []
    dpdt = []

    for i in range(1, len(a_peaks) - 1):
        itv = up_peaks[i] - a_peaks[i]
        if a_peaks[i] - itv * 1.6 < 0:
            continue
        # aitv = int((a_peaks[i + 1] - a_peaks[i])// 3)
        S1 = a_peaks[i] - int(itv * 1.6) + np.argmax(pcg[a_peaks[i] - int(itv * 1.6):a_peaks[i]])
        S2 = up_peaks[i] - int(itv * 0.05) + np.argmax(pcg[up_peaks[i] - int(itv * 0.05):up_peaks[i] + 150])
        s1_index.append(S1)
        s2_index.append(S2)
        Psys.append(up_peaks[i])
        Pdias.append(a_peaks[i])
        s12i.append((S2 - S1) / HZ)
        itv2.append((S2 - a_peaks[i]) / HZ)

    return s1_index, s2_index, s12i, Psys, Pdias, itv2, dpdt



def PCG_detect_abp_v2(a_peaks, up_peaks, pcg, HZ):
    sink = 0
    while up_peaks[sink] < a_peaks[0]:
        sink += 1
        if sink >= len(up_peaks):
            break
    dp = []
    up_peaks = up_peaks[sink:]
    for i in range(len(up_peaks)):
        for j in range(len(a_peaks)):
            pitv = up_peaks[i] - a_peaks[j]
            if pitv < 0:
                dp.append(a_peaks[j - 1])
                break
    a_peaks = dp
    Psys = []
    Pdias = []
    pp = []
    s1_index = []
    s2_index = []
    s12i = []
    itv2 = []
    dpdt = []

    for i in range(1, len(a_peaks) - 1):
        itv = up_peaks[i] - a_peaks[i]
        if a_peaks[i] - itv * 1.6 < 0:
            continue
        # aitv = int((a_peaks[i + 1] - a_peaks[i])// 3)
        S1 = a_peaks[i] - int(itv * 1.6) + np.argmax(pcg[a_peaks[i] - int(itv * 1.6):a_peaks[i]])
        S2 = up_peaks[i] - int(itv * 0.05) + np.argmax(pcg[up_peaks[i] - int(itv * 0.05):up_peaks[i] + 150])
        s1_index.append(S1)
        s2_index.append(S2)
        Psys.append(up_peaks[i])
        Pdias.append(a_peaks[i])
        s12i.append((S2 - S1) / HZ)
        itv2.append((S2 - a_peaks[i]) / HZ)

    abp_rr = []
    for i in range(1,len(a_peaks)):
        abp_rr.append(a_peaks[i] - a_peaks[i-1])

    med_rr = np.median(abp_rr)

    abp_rr = []
    s21int = []
    per_s12,    per_s21= [],[]

    for i in range(1,len(s2_index)):
        s21 = s1_index[i] -s2_index[i-1]
        rr = a_peaks[i] - a_peaks[i-1]
        #check rr 2 beat
        #print(rr,med_rr*0.5)
        if rr > med_rr*1.5 :
            continue
        s21int.append(s21)
        abp_rr.append(rr)
        per_s12.append((rr-s21)/rr)
        per_s21.append(s21/rr)





    return s1_index, s2_index, s12i, Psys, Pdias, s21int,abp_rr,per_s12,per_s21

def PCG_detect_abp_dpdt(a_peaks, up_peaks, pcg, abp, HZ):
    sink = 0
    while up_peaks[sink] < a_peaks[0]:
        sink += 1
        if len(sink) == sink:
            break
    dp = []
    up_peaks = up_peaks[sink:]
    for i in range(len(up_peaks)):
        for j in range(len(a_peaks)):
            pitv = up_peaks[i] - a_peaks[j]
            if pitv < 0:
                dp.append(a_peaks[j - 1])
                break
    a_peaks = dp
    Psys = []
    Pdias = []
    pp = []
    s1_index = []
    s2_index = []
    s12i = []
    itv2 = []
    dpdt = []
    dpdt_wave = np.diff(arr.band_pass(abp, HZ, 1, 20))
    dpdt_wave[0:HZ // 10] = 0
    dpdt_wave[len(dpdt) - HZ // 10:] = 0

    for i in range(1, len(a_peaks) - 1):
        itv = up_peaks[i] - a_peaks[i]
        if a_peaks[i] - itv * 1.6 < 0:
            continue
        # aitv = int((a_peaks[i + 1] - a_peaks[i])// 3)
        S1 = a_peaks[i] - int(itv * 1.6) + np.argmax(pcg[a_peaks[i] - int(itv * 1.6):a_peaks[i]])
        S2 = up_peaks[i] - int(itv * 0.05) + np.argmax(pcg[up_peaks[i] - int(itv * 0.05):up_peaks[i] + int(itv * 1.7)])
        s1_index.append(S1)
        s2_index.append(S2)
        Psys.append(up_peaks[i])
        Pdias.append(a_peaks[i])
        s12i.append((S2 - S1) / HZ)
        itv2.append((S2 - a_peaks[i]) / HZ)
        # dpdt.append(np.max(dpdt_wave[a_peaks[i] - int(itv * 0.5):a_peaks[i] + int(itv * 0.5)]))

    return s1_index, s2_index, s12i, Psys, Pdias, itv2, dpdt


def check_abp_int(a_peaks, up_peaks):
    sink = 0
    while up_peaks[sink] < a_peaks[0]:
        sink += 1
        if sink >= len(up_peaks):
            break
    dp = []
    up_peaks = up_peaks[sink:]
    for i in range(len(up_peaks)):
        for j in range(len(a_peaks)):
            pitv = up_peaks[i] - a_peaks[j]
            if pitv < 0:
                dp.append(a_peaks[j - 1])
                break
    a_peaks = dp

    for i in range(len(a_peaks) - 1):
        if up_peaks[i] - a_peaks[i] > a_peaks[i + 1] - up_peaks[i]:
            return True
    return False


def check_abp_fft(abp):
    fttt = np.fft.fft(abp)
    if np.max(np.diff(fttt[50:60])) > 5000:
        return True
    return False


def check_PCG_start(PCG, HZ):
    import numpy as np
    LEN_UNIT_DATASET = HZ * 60 * 15  # unit length of Peak detection 15min :
    st = -1

    for k in range(len(PCG) // LEN_UNIT_DATASET - 1):
        pcg = PCG[k * LEN_UNIT_DATASET:k * LEN_UNIT_DATASET + HZ * 20]
        pcg2 = PCG[(k + 1) * LEN_UNIT_DATASET:(k + 1) * LEN_UNIT_DATASET + HZ * 20]
        if np.std(pcg) > 0.15 and np.std(pcg2) > 0.15:
            st = k * LEN_UNIT_DATASET
            break
    if st == -1:
        return -1, -1
    for k in reversed(range(len(PCG) // LEN_UNIT_DATASET)):
        pcg = PCG[k * LEN_UNIT_DATASET: k * LEN_UNIT_DATASET + HZ * 20]
        pcg2 = PCG[(k - 1) * LEN_UNIT_DATASET:(k - 1) * LEN_UNIT_DATASET + HZ * 20]
        if np.std(pcg) > 0.15 and np.std(pcg2) > 0.15:
            et = k * LEN_UNIT_DATASET
            break
    return st, et


def get_variation(data):
    data1 = data[0:len(data) // 3]
    data2 = data[len(data) // 3:len(data) * 2 // 3]
    data3 = data[len(data) * 2 // 3:len(data)]
    v1 = (np.max(data1) - np.min(data1)) / (np.mean(data1))
    v2 = (np.max(data2) - np.min(data2)) / (np.mean(data2))
    v3 = (np.max(data3) - np.min(data3)) / (np.mean(data3))
    return np.mean([v1, v2, v3])


def search_fullfile(path):
    file = os.listdir(path)
    filenames = []
    for filename in file:
        file_path = path[0:28]
        File_date = filename[-19:-13]
        full_filename = file_path + File_date + "/" + filename
        if (os.path.getsize(full_filename) < 50000000):
            print("small file")
            continue
        filenames.append(full_filename)
    return filenames


# draw graph
def total_draw_corr(TARGET_PATH, FILE_DATE, DATE_TIME, TITLE, live, datetime, data1, data1name, data2, data2name):
    fig = plt.figure(figsize=(20, 10))
    ax2 = fig.subplots()
    # interval and pp corr
    # interval and pp corr
    ax2.plot(datetime, data1, marker='s', label=data1name)
    cor = np.corrcoef(data1, data2)
    stcor = str(round(cor[0, 1], 2))
    plt.legend(loc='upper left')
    ax2 = ax2.twinx()
    ax2.plot(datetime, data2, 'r', marker='s', label=data2name)
    plt.title(TITLE + ", correlation:" + stcor + ' data:' + live)
    plt.legend(loc='upper right')
    plt.savefig(
        TARGET_PATH + "/" + FILE_DATE + "/" + "total_result/" + FILE_DATE + "_" + DATE_TIME + "_" + TITLE + ".png")
    plt.close()
    return stcor


def total_draw_corr_event(TARGET_PATH, FILE_DATE, DATE_TIME, TITLE, live, datetime, data1, data1name, data2, data2name,
                          eventtime):
    fig = plt.figure(figsize=(20, 10))
    ax2 = fig.subplots()
    # interval and pp corr
    # interval and pp corr
    ax2.plot(datetime, data1, marker='s', label=data1name)
    cor = np.corrcoef(data1, data2)
    stcor = str(round(cor[0, 1], 2))
    plt.legend(loc='upper left')
    ax2 = ax2.twinx()
    ax2.plot(datetime, data2, 'r', marker='s', label=data2name)
    eventpoint = np.mean(data2)

    ax2.annotate('IVC Clamping', xy=(eventtime[0], eventpoint), xytext=(eventtime[0], eventpoint),
                 arrowprops=dict(facecolor='black', shrink=0.2))
    ax2.annotate('Declamping', xy=(eventtime[1], eventpoint), xytext=(eventtime[1], eventpoint),
                 arrowprops=dict(facecolor='black', shrink=0.2))

    plt.title(TITLE + ", correlation:" + stcor + ' data:' + live)
    plt.legend(loc='upper right')
    plt.savefig(
        TARGET_PATH + "/" + FILE_DATE + "/" + "total_result/" + FILE_DATE + "_" + DATE_TIME + "_" + TITLE + ".png")
    plt.close()
    return stcor


def sca_total_draw_corr(TARGET_PATH, FILE_DATE, DATE_TIME, TITLE, live, datetime, data1, data1name, data2, data2name):
    fig, ax2 = plt.subplots()
    # interval and pp corr
    ax2.scatter(datetime, data1, label=data1name, s=20)
    cor = np.corrcoef(data1, data2)
    stcor = str(round(cor[0, 1], 2))
    plt.legend(loc='upper left')
    ax2 = ax2.twinx()
    ax2.scatter(datetime, data2, color='RED', label=data2name, s=20)
    plt.title(TITLE + ", correlation:" + stcor + " data:" + live)
    plt.legend(loc='upper right')
    plt.savefig(
        TARGET_PATH + "/" + "total_result/" + FILE_DATE + "/" + FILE_DATE + "_" + DATE_TIME + "_" + TITLE + ".png")
    plt.close()
    return stcor


def draw_corr(TARGET_PATH, FILE_DATE, DATE_TIME, TITLE, datetime, data1, data1name, data2, data2name):
    fig, ax2 = plt.subplots(figsize=(20, 10))
    # interval and pp corr
    ax2.plot(datetime, data1, marker='s', label=data1name)
    cor = np.corrcoef(data1, data2)
    stcor = str(round(cor[0, 1], 2))
    plt.legend(loc='upper left')

    ax2 = ax2.twinx()
    ax2.plot(datetime, data2, 'r', marker='s', label=data2name)

    plt.title(TITLE + ", correlation:" + stcor)
    plt.legend(loc='upper right')
    plt.savefig(TARGET_PATH + "/" + FILE_DATE + "/" + FILE_DATE + "_" + DATE_TIME + "_" + TITLE + ".png")
    plt.close()
    return stcor


def draw_corr_1beat_move(TARGET_PATH, FILE_DATE, DATE_TIME, TITLE, datetime, data1, data1name, data2, data2name):
    fig, ax2 = plt.subplots(figsize=(20, 10))
    # interval and pp corr
    ax2.plot(datetime[0:-1], data1[0:-1], marker='s', label=data1name)
    cor = np.corrcoef(data1[0:-1], data2[0:-1])
    stcor = str(round(cor[0, 1], 2))
    plt.legend(loc='upper left')

    ax2 = ax2.twinx()
    ax2.plot(datetime[0:-1], data2[1:], 'r', marker='s', label=data2name)

    plt.title(TITLE + ", correlation:" + stcor)
    plt.legend(loc='upper right')
    plt.savefig(TARGET_PATH + "/" + FILE_DATE + "/" + FILE_DATE + "_" + DATE_TIME + "_" + TITLE + ".png")
    plt.close()
    return stcor


def draw_corr_other(TARGET_PATH, FILE_DATE, DATE_TIME, TITLE, datetime, data1, data1name, data2, data2name):
    fig, ax2 = plt.subplots(figsize=(20, 10))
    # interval and pp corr
    ax2.plot(data1, marker='s', label=data1name)
    cor = np.corrcoef(data1, data2)
    stcor = str(round(cor[0, 1], 2))
    plt.legend(loc='upper left')

    ax2 = ax2.twinx()
    ax2.plot(data2, 'r', marker='s', label=data2name)
    plt.title(TITLE + ", correlation:" + stcor)
    plt.legend(loc='upper right')
    plt.savefig(TARGET_PATH + "/" + FILE_DATE + "/" + FILE_DATE + "_" + DATE_TIME + "_" + TITLE + ".png")
    plt.close()
    return stcor
