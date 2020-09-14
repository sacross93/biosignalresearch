# -*- coding: euc-kr -*-
import os


# ��¥�� �޾Ƽ� ��¥���� ������ ������ True, ������ false
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
    # ���� ��ġ ��� �ľ�
    default_path = "/mnt/Data/CloudStation"

    # ���� ������� ���
    data = {}
    result = set()
    # ������ �ִ� ���� ������ ������ Ȯ������.
    for today in days:
        for i in roomname:
            data[i] = search(today, default_path + "/" + i)
            if data[i] == True:
                filename = os.listdir(default_path + "/" + i + "/" + today)
                for j in filename:

                    fullfilename = default_path + "/" + i + "/" + today + "/" + j
                    filesize = os.path.getsize(fullfilename)
                    # filesize üũ �� check���α׷� ����.
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
