import os, csv
import pandas as pd
import re

# 실습1
# Header 확인하기 + row 개수 세기

# os.chdir('c://py_work')
os.chdir('C://Users/jykim/Downloads/Week2/3. EMR 데이터 심화실습/2021 EMR 심화실습_자료')

f = open('lab_result_sample.csv', 'r', encoding='cp949')
rdr = csv.reader(f)

row_count = 0

for x in rdr:
    if row_count == 0:
        print(x)
    else:
        pass
    row_count += 1

print(str(row_count))
f.close()

# 이렇게 쓰는 것이 훨씬 효율적임
data = pd.read_csv('lab_result_sample.csv',encoding='cp949')
print(len(data))

#-------------------------------------------

# lab_code가 L3098인 검사 결과만 새로운 파일로 만들기
import os, csv, re

# os.chdir('c://py_work')

f = open('lab_result_sample.csv', 'r', encoding='cp949')

rdr = csv.reader(f)
next(rdr)

ff = open('L3098_new.csv', 'w', newline='')
wr = csv.writer(ff)

for x in rdr:
    lab_code = x[2]
    if lab_code == "L3098":
        wr.writerow(x)
    else:
        pass
f.close()
ff.close()

# 정규표현식 (주민번호)
# (\d{2})(\d{2})(\d{2}) - (1 | 2)(\d{6})

# 문자열 매치하기
import re

text_ex = "1234 123"
m = re.match('[0-9]', text_ex)
print(m.group())
m = re.match('[0-9]+', text_ex)
print(m.group())
m = re.match('[0-9]+\s[0-9]+', text_ex)
print(m.group())
m = re.search('\s[0-9]+', text_ex)
print(m.group())
m = re.match('\d+\s\d+', text_ex)
print(m.group())
# 다른 methods를 활용하여 입력해봅시다.


# 문자열 추출하기
my_phone = "My name is Yura Lee and my phone number is 010-2222-7777 \n and my e-mail is haepary@naver.com"
m = re.search("name is (.*) and", my_phone)
print(m.group(1))

m = re.search(r"(\d+)-(\d+)-(\d+)", my_phone)
print(m.group(1) + '-' + m.group(2) + '-' + m.group(3))

m = re.search(r'01[01]-[\d]{3,4}-[\d]{4}', my_phone)
print(m.group())

m = re.search(r'(\S+@\S+)', my_phone)
print(m.group())

# 문자열 내용을 리스트로 만들기
sample_text = 'I want to go home : I want to sleep'
m = re.split(':', sample_text)

print(m)

# 실습2


# os.chdir('c://py_work')/
# PC에서 작업할 경우

f = open('lab_result_sample.csv', 'r', encoding='cp949')
rdr = csv.reader(f)
next(rdr)

ff = open('L3098.csv', 'w', newline='')
wr = csv.writer(ff)

for x in rdr:
    lab_code = x[2]
    lab_result = x[4]
    if lab_code == "L3098":
        # 세 번째 컬럼 (lab_code) 값이 L3098인 값만 추출하여 새로운 CSV 파일로 써주는 것
        m = re.match(r'(\d+)', lab_result)
        if m:
            wr.writerow(x)
        # 결과값 (다섯 번째 컬럼)이 숫자로만 되어있을 경우는 값을 바로 새로운 CSV에 써줌
        else:
            mm = re.search(r'(\d+)\D+(\d+)', lab_result)
            if mm:
                x[4] = mm.group(2)
                wr.writerow(x)
            # 결과값에 숫자와 문자가 섞여있는 경우, 두 번째 숫자 그룹의 값을 추출해 새로운 CSV에 써줌
            else:
                pass

f.close()
ff.close()

# CSV 읽고 쓰기
# dataframe_name2=csv
# dataframe_name1 = pd.read_csv('filename1.csv')  # 첫 행이 header가 아니면 header = None
# dataframe_name2.to_csv('filename2.csv')  # defalut: header = True (첫번째 줄을 칼럼 이름으로 사용)


# Google colab에서 파일 읽어오기
# from google.colab import files

# myfile = files.upload()
# 내 하드웨어에서 파일선택
import io  # pandas를 불러오지 않았을 경우 “import pandas as pd”도 같이 해줍시다.

df1 = pd.read_csv(io.BytesIO(os.path('DATA_B.csv')))
df1.head(5)

# from google.colab import drive

# drive.mount('/content/drive’)
filename = 'DATA_B2.csv' # 우클릭후 파일 경로 확인
df2 = pd.read_csv(filename)  # 변수 (filename) 대신에 바로 파일 경로 입력 가능

url = 'https://raw.githubusercontent.com/haepary/lecture_data/master/lab_result_sample.csv'
# 파일경로
df3 = pd.read_csv(url, encoding='cp949')

# 실습3 유방초음파 소견에서 유방암으로 확인된 환자 ID 추출하기

import os, re

os.chdir('c://py_work')

in_text = open("breast_us_sample.txt", "r")
out_text = open("breast_cancer_out.txt", "w")

PT_cat = {}

for x in in_text:
    m = re.search(r'(\D{3}\d{3})', x)
mm = re.search(r'BIRADS\scategory\s([A-Z]+)', x)
if m:
    PT_ID = m.group(1)
elif mm:
    cat_no = mm.group(1)
PT_cat[PT_ID] = cat_no
else:
pass

for x in PT_cat:
    if
PT_cat[x] == "V"
OR
PT_cat[x] == "VI":
print(x)
else:
pass

# 실습4 SEER DB에서 특정기간 유방암 여성 환자 수 확인하기

import os, re

os.chdir('c://py_work')

in_text = open("seer_breast_sample.txt", "r")
out_text = open("breast_enrol_sample.txt", "w")

total_PT = 0
f_count = 0
m_count = 0
Dx_year_count = 0
stage_count = 0

for x in in_text:
    x_ID = x[0:8]
x_year = int(x[38:42])
x_sex = int(x[23:24])
x_stage7 = x[320:323]
x_stage6 = x[329:331]
total_PT = total_PT + 1
if x_sex == 2:
    f_count = f_count + 1
if x_year >= 2000 and x_year <= 2015:
    Dx_year_count = Dx_year_count + 1
m = re.match("\d", x_stage7)
if m:
    int_stage7 = int(x_stage7)
if int_stage7 >= 100 and int_stage7 < 700:
    stage_count = stage_count + 1
out_text.write(x)
else:
pass
else:
m = re.match("\d", x_stage6)
if m:
    int_stage6 = int(x_stage6)
if int_stage6 >= 10 and int_stage6 < 70:
    stage_count = stage_count + 1
out_text.write(x)
else:
pass
elif x_sex == 1:
m_count = m_count + 1
else:
pass

print(total_PT, f_count, m_count, Dx_year_count, stage_count)

in_text.close()
out_text.close()

# 데이터프레임만들기
import pandas as pd
import numpy as np

list_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
pd.DataFrame(list_2d)

my_dic = {'a': [1, 3], 'b': [2, 4], 'c': [5, 6]}
pd.DataFrame(my_dic)

pd.Series(['i', 'am', 'hungry’])

           menu_name = pd.Series(['coffee', 'muffin', 'juice'])
prices = pd.Series([3000, 2500, 5500])
pd.DataFrame({"Menu names": menu_name, "Prices": prices})

# 데이터프레임
df0 = pd.DataFrame(my_dic)
df0.loc[1] = [7, 9, 8]
# df붙이기 df0 = df0.append(df00)
df0[‘d’]=pd.Series([10, 12], index=df0.index)

# 실습5
import matplotlib.pyplot as plt

df4 = pd.read_csv('/L3098.csv')  # df4.head() 입력해서 확인
df4.columns = ['ID', 'date', 'lab_code', 'code_name', 'result']  # df4.head()
df5 = pd.read_csv('/L3098.csv', names=['ID', 'date', 'lab_code', 'code_name', 'result’])    #df5.head()
                                       # 데이터프레임 csv로 저장하기: df5.to_csv('L3098_header.csv')
                                       # 구글드라이브의 저장경로 지정하기: !cp L3098_header.csv '저장경로'

                                       plt.hist(df5['result'], bins=10)  # bins 값을 조절해가며 결과 보기


# 생존분석
import?pandas?as?pd
import?numpy?as?np
import?matplotlib.pyplot?as?plt
from?lifelines?import?KaplanMeierFitter
durations?=?[5, 6, 6, 2.5, 4, 4]
event_observed?=?[1, 0, 0, 1, 1, 1]
kmf?=?KaplanMeierFitter()
kmf.fit(durations,?event_observed,?label = 'Kaplan?Meier?Estimate')
kmf.plot(ci_show=False)  # Confidence interval은 안 봅니다.


# 시간데이터
import pandas as pd
from datetime import datetime, timedelta

a = "2021-03-21“
A = datetime.strptime(a, '%Y-%m-%d’)
b = "21-May-21"
B = datetime.strptime(b,’ % y - % B - % d’)  # %y는 두 자리 연도, %B는 영문 월 이름에 연결
C = B ? A
Print(C)

import pandas as pd
from datetime import datetime, timedelta

df1 = pd.read_csv('/content/DATA_B1.csv’)
df1.head()

Df1.info()
print(df1['Dx_age'].hist())

df1['Dx_date'] = pd.to_datetime(df1['Dx_date'])
df1['final_status_date'] = pd.to_datetime(df1['final_status_date'])

df1['date_delta'] = df1[‘final_status_date
'] ? df1['
Dx_date’]

Df1.info()

# 생존분석
kmf = KaplanMeierFitter()
kmf.fit(durations=df1["date_delta"], event_observed=df1["dead"])

kmf.plot()
plt.title("The Kaplan-Meier Estimate")
plt.xlabel("Number of days")
plt.ylabel("Probability of survival")

# 생존분석: Cox regression
df2 = pd.read_csv('/content/DATA_B2.csv’)
df2.head()

df2.info()
print(df2[seer_stage
'].hist())

from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(df2, duration_col='date_delta', event_col='dead’)

cph.print_summary()

cph.plot()

cph.plot_partial_effects_on_outcome(covariates='seer_stage', values=[0, 1, 2, 3, 4, 7, 9], cmap='coolwarm')

# 끝!