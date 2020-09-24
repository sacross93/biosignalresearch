import vitaldb
import numpy as np
import pandas as pd

LSTM_TIMEPOINTS = 180
LSTM_NODES = 8
FNN_NODES = 16
BATCH_SIZE = 256  # 한번에 처리할 레코드 수 (GPU 메모리 용량에 따라 결정)
MAX_CASES = 50  # 본 예제에서 사용할 최대 case 수

df_trks = pd.read_csv('https://api.vitaldb.net/v2/trks')
ppf_cases = set(df_trks[df_trks['tname'] == 'Orchestra/PPF20_VOL']['caseid'])  # print(len(ppf_cases))  # 3523
rft_cases = set(df_trks[df_trks['tname'] == 'Orchestra/RFTN20_VOL']['caseid'])  # print(len(rft_cases))  # 4791
tiva_cases = ppf_cases & rft_cases  # print(len(tiva_cases))  # 3457
bis_cases = set(df_trks[df_trks['tname'] == 'BIS/BIS']['caseid'])  # print(len(bis_cases))  # 5867
study_cases = sorted(list(tiva_cases & bis_cases))  # print(len(study_cases))  # 3289

df_cases = pd.read_csv("https://api.vitaldb.net/cases")
cases = df_cases.loc[[caseid in study_cases for caseid in df_cases['caseid']], ['caseid', 'age', 'sex', 'weight', 'height']]
cases['sex'] = cases['sex'] == 'F'
cases = cases[cases['age'] > 18]
cases = cases[cases['weight'] > 35]

caseids = cases.values[:,0]
caseid_aswh = {row[0]: row[1:].astype(float) for row in cases.values}
np.random.shuffle(caseids)

# vital 파일로부터 dataset 을 만듬
x_ppf = []  # 각 레코드의 프로포폴 주입량
x_rft = []  # 각 레코드의 레미펜타닐 주입량
x_aswh = []  # 각 레코드의 나이, 성별, 키, 몸무게
x_caseid = []  # 각 레코드의 caseid
y = []  # 각 레코드의 출력값 (bis)


PPF_DOSE = 0
RFT_DOSE = 1
BIS = 2
icase = 0
ncase = min(MAX_CASES, len(caseids))
for caseid in caseids:
    ppf20_tid = df_trks[(df_trks['caseid'] == caseid) & (df_trks['tname'] == 'Orchestra/PPF20_VOL')]['tid'].values[0]
    rftn20_tid = df_trks[(df_trks['caseid'] == caseid) & (df_trks['tname'] == 'Orchestra/RFTN20_VOL')]['tid'].values[0]
    bis_tid = df_trks[(df_trks['caseid'] == caseid) & (df_trks['tname'] == 'BIS/BIS')]['tid'].values[0]

    vals = vitaldb.load_trks([ppf20_tid, rftn20_tid, bis_tid], 10) #10s 간격 추출

    # 2시간 이내의 case 들은 사용하지 않음, 720 =  2hr * 60min/hr * 6개/min
    if vals.shape[0] < 720:
        continue

    # 결측값은 측정된 마지막 값으로 대체
    vals = pd.DataFrame(vals).fillna(method='ffill').values
    vals = np.nan_to_num(vals, 0)  # 맨 앞 쪽 결측값은 0으로 대체

    # drug 주입을 하지 않은 경우 혹은 bis를 켜지 않은 경우 사용하지 않음
    if (np.max(vals, axis=0) <= 1).any():
        continue

    # drug infusion 시작 시간을 구하고 그 이전을 삭제
    first_ppf_idx = np.where(vals[:, PPF_DOSE] > 1)[0][0]
    first_rft_idx = np.where(vals[:, RFT_DOSE] > 1)[0][0]
    first_drug_idx = min(first_ppf_idx, first_rft_idx)
    vals = vals[first_drug_idx:, :]

    # volume 을 rate로 변경
    vals[1:, PPF_DOSE] -= vals[:-1, PPF_DOSE]
    vals[1:, RFT_DOSE] -= vals[:-1, RFT_DOSE]
    vals[0, PPF_DOSE] = 0
    vals[0, RFT_DOSE] = 0

    # 음수 값(volume 감소)을 0으로 대체
    vals[vals < 0] = 0

    # 유효한 bis 값들을 가져옴
    valid_bis_idx = np.where(vals[:, BIS] > 0)[0]

    # 2시간 이내의 case들은 사용하지 않음
    if valid_bis_idx.shape[0] < 720:
        continue

    # bis 값의 첫 값이 80 이하이거나 마지막 값이 70 이하인 case는 사용하지 않음
    first_bis_idx = valid_bis_idx[0]
    last_bis_idx = valid_bis_idx[-1]
    if vals[first_bis_idx, BIS] < 80 or vals[last_bis_idx, BIS] < 70:
        continue

    icase += 1
    print('loading ({}/{}): caseid {:.0f}, first bis {:.1f}, last bis {:.1f}'.format(icase, ncase, caseid, vals[first_bis_idx, BIS], vals[last_bis_idx, BIS]))

    # infusion 시작 전 LSTM_TIMEPOINTS 동안의 dose와 bis를 모두 0으로 세팅
    vals = np.vstack((np.zeros((LSTM_TIMEPOINTS - 1, 3)), vals))

    # 현 case의 나이, 성별, 키, 몸무게를 가져옴
    aswh = caseid_aswh[caseid]

    # case 시작 부터 종료 까지 dataset 에 넣음
    for irow in range(1, vals.shape[0] - LSTM_TIMEPOINTS - 1):
        bis = vals[irow + LSTM_TIMEPOINTS, BIS]
        if bis == 0:
            continue

        # 데이터셋에 입력값을 넣음
        x_ppf.append(vals[irow:irow + LSTM_TIMEPOINTS, PPF_DOSE])
        x_rft.append(vals[irow:irow + LSTM_TIMEPOINTS, RFT_DOSE])
        x_aswh.append(aswh)
        x_caseid.append(caseid)
        y.append(bis)

    if icase >= ncase:
        break

# 모든 케이스 확인, xppf... y까지 담은후,
# 입력 데이터셋을 numpy array로 변경

x_ppf = np.array(x_ppf)[..., None]  # LSTM 에 넣기 위해서는 3차원이어야 한다. 마지막 차원을 추가
x_rft = np.array(x_rft)[..., None]
x_aswh = np.array(x_aswh)
y = np.array(y)
x_caseid = np.array(x_caseid)

# 최종적으로 로딩 된 caseid
caseids = np.unique(x_caseid)

# normalize data
x_aswh = (x_aswh - np.mean(x_aswh, axis=0)) / np.std(x_aswh, axis=0)

# bis 값은 최대값이 98 이므로 98로 나눠 normalization
y /= 98

# train, test case로 나눔
ntest = int(ncase * 0.1)
ntrain = ncase - ntest
train_caseids = caseids[:ntrain]
test_caseids = caseids[ntrain:ncase]

# train set과 test set 으로 나눔
train_mask = np.array([caseid in train_caseids for caseid in x_caseid])
test_mask = np.array([caseid in test_caseids for caseid in x_caseid])
x_train = [x_ppf[train_mask], x_rft[train_mask], x_aswh[train_mask]]
y_train = y[train_mask]
x_test = [x_ppf[test_mask], x_rft[test_mask], x_aswh[test_mask]]
y_test = y[test_mask]

print('train: {} cases {} samples'.format(len(train_caseids), np.sum(train_mask)))
print('test: {} cases {} samples'.format(len(test_caseids), np.sum(test_mask)))

from keras.models import Model, load_model
from keras.layers import Dense, Dropout, LSTM, Input, concatenate
from keras.callbacks import EarlyStopping
import tensorflow as tf

# 모델 설계
input_cov = Input(batch_shape=(None, 4))
input_ppf = Input(batch_shape=(None, LSTM_TIMEPOINTS, 1))
input_rft = Input(batch_shape=(None, LSTM_TIMEPOINTS, 1))
output_ppf = LSTM(LSTM_NODES, input_shape=(LSTM_TIMEPOINTS, 1), activation='tanh')(input_ppf)
output_rft = LSTM(LSTM_NODES, input_shape=(LSTM_TIMEPOINTS, 1), activation='tanh')(input_rft)
output = concatenate([output_ppf, output_rft, input_cov])
output = Dense(FNN_NODES)(output)
output = Dropout(0.2)(output)
output = Dense(1, activation='sigmoid')(output)

mae = tf.keras.losses.MeanAbsoluteError()
mape = tf.keras.losses.MeanAbsolutePercentageError()

model = Model(inputs=[input_ppf, input_rft, input_cov], outputs=[output])
model.compile(loss=mae, optimizer='adam', metrics=[mape])
hist = model.fit(x_train, y_train, validation_split=0.1, epochs=100, steps_per_epoch=100,
                           callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')])

# 출력 폴더를 생성
import os
odir = 'output'
if not os.path.exists(odir):
    os.mkdir(odir)

y_pred = model.predict(x_test).flatten()
test_mape = mape(y_test, y_pred)

print("Test MAPE: {}".format(test_mape))

# 각 case에서 예측 결과를 그림으로 확인
import matplotlib.pyplot as plt
for caseid in test_caseids:
    case_mask = (x_caseid[test_mask] == caseid)
    case_len = np.sum(case_mask)
    if case_len == 0:
        continue

    case_mape = mape(y_test[case_mask], y_pred[case_mask])
    print('{}\t{}'.format(caseid, case_mape))

    t = np.arange(0, case_len)
    plt.figure(figsize=(20, 5))
    plt.plot(t, y_test[case_mask], t, y_pred[case_mask])
    plt.xlim([0, case_len])
    plt.ylim([0, 1])
    plt.savefig('{}/{:.3f}_{}.png'.format(odir, case_mape, caseid))
    plt.close()

os.rename(odir, 'res {}'.format(test_mape))

