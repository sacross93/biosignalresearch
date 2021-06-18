from datetime import datetime


def replace_sharp(s):
    return s.replace("C#", 'c').replace("D#", 'd').replace("F#", 'f').replace("G#", 'g').replace("A#", 'a')


def slice_str_list(musicinfos):
    slice_musicinfos = []
    for i in musicinfos:
        slice_musicinfos.append(i.split(','))
    date_formatter = "%H:%M"
    for i in range(0, len(slice_musicinfos)):
        slice_musicinfos[i][0] = datetime.strptime(slice_musicinfos[i][0], date_formatter)
        slice_musicinfos[i][1] = datetime.strptime(slice_musicinfos[i][1], date_formatter)

    for i in range(0, len(slice_musicinfos)):
        for j in range(0, len(slice_musicinfos)):
            slice_musicinfos[j][3] = replace_sharp(slice_musicinfos[j][3])

        originally_melody = slice_musicinfos[i][3]
        minute = (((slice_musicinfos[i][1] - slice_musicinfos[i][0]).total_seconds()) % 3600) // 60
        musicinfos2 = slice_musicinfos[i][3] * (int(minute))
        slice_musicinfos[i][3] = musicinfos2[:int(minute)]

    for i in range(0, len(slice_musicinfos)):
        del slice_musicinfos[i][1]
        del slice_musicinfos[i][0]
    return slice_musicinfos


def is_it_answer(musicinfos, m):
    if musicinfos[:len(m)] == m: return True
    return False


def solution(m, musicinfos):
    answer = ''
    real_m = replace_sharp(m)
    slice_musicinfos = slice_str_list(musicinfos)

    for i in range(0, len(slice_musicinfos)):
        for j in range(0, len(slice_musicinfos[i][1])):
            if real_m[0] == slice_musicinfos[i][1][j]:
                if is_it_answer(slice_musicinfos[i][1][j:], real_m):
                    answer = slice_musicinfos[i][0]
                    return answer
    if not answer: return '(None)'