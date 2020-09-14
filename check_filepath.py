# -*- coding: euc-kr -*-
import os


# 날짜를 받아서 날짜폴더 파일이 있으면 True, 없으면 false
def search(today, dirname):
    flag = False
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        # print(full_filename)
        if full_filename == (dirname + "/" + today):
            flag = True
    return flag


def search_filepath(roomname, days):
    # 현재 위치 경로 파악
    default_path = "/mnt/Data/CloudStation"

    # 현재 사용중인 방들
    data = {}
    result = set()
    # 폴더가 있는 방의 데이터 전송을 확인해줌.
    for today in days:
        for i in roomname:
            data[i] = search(today, default_path + "/" + i)
            if data[i] == True:
                filename = os.listdir(default_path + "/" + i + "/" + today)
                for j in filename:

                    fullfilename = default_path + "/" + i + "/" + today + "/" + j
                    filesize = os.path.getsize(fullfilename)
                    # filesize 체크 후 check프로그램 돌림.
                    if filesize / 100.0 ** 2 > 1:
                        result.add(fullfilename)
    return result


"""
roomname = [
            "J-01","J-02","J-03","J-04","J-05","J-06",
            ]

days = ["170501","180201"]

result = search_filepath(roomname,days)

print(result)
"""
