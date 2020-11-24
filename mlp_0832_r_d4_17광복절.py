# -*- coding: utf-8 -*-
"""MLP_0832_R_d4_17광복절.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hooiKVO-2WaRz3FxrEhxwhQUNuERozYs
"""

from google.colab import files
upload1 = files.upload()

SEQ_LEN = 1
FUTURE_PERIOD_PREDICT = 1

# Parameters setting 
#========================================
BATCH_SIZE = 64
TITLE = 'Independence'
l1 = 8
l2 = 16
#index_n = -3 

#plust one 
date_MDDYYYYVTTTT = '8/16/2017 00:00' 
#========================================
import tensorflow as tf

import numpy as np
import pandas as pd
from sklearn import preprocessing
from collections import deque

from sklearn import model_selection

import random
from time import gmtime, strftime
from datetime import datetime
import itertools
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM,Activation, Dropout, BatchNormalization

import matplotlib.pyplot as plt
import seaborn as sns

import io
import pandas as pd 
dataset = pd.read_csv(io.BytesIO(upload1['total_00_180414.csv']), index_col = 0)

#d1 = ['month', 'date','hour','day','elec']
#d2 = ['ss_all','month', 'date','hour','day','elec']
d4 = ['ss_all', 'temp_tt','humid_tt','day', 'wind_tt','rainfall_tt','elec']
data = dataset.copy()
dt = data[d4]

display(dt.head(2))
display(dt.tail(2))

# dt = pd.get_dummies(dt, columns=['month', 'date','hour','day'])
dt = pd.get_dummies(dt, columns=['day'])
display(dt.head(2))
display(dt.tail(2))

def to_float(dt):
  pm3 = dt
  iter_cols = ['temp_tt', 'humid_tt', 'wind_tt', 'rainfall_tt', 'elec']
  for i in iter_cols:
    pm3[i] = pm3[i].astype(float)
  return pm3 

dt = to_float(dt)

def norm(dt):
  pm3 = dt
  scaler = preprocessing.MinMaxScaler()
  pm3['elec'] = pm3['elec']/float(100000.0)
  pm3['shift_1'] = pm3['elec'].shift(24)
  pm3['shift_7'] = pm3['elec'].shift(168)
  dt = pm3
  dt.dropna(inplace = True)
  return dt

dt = norm(dt)
dt.head(2)

def reorder(dt):
  
  cols = dt.columns.tolist()
  cols.remove('elec')
  cols.append('elec')
  print(cols)
  dt = dt[cols]
  return dt

dt= reorder(dt)

display(dt.head(2))
display(dt.tail(2))

dt.head(3)

def cut_end_set(pm3, date_MDDYYYYVTTTT):
  # special DAY1 00:00 
  return pm3.loc[:date_MDDYYYYVTTTT]

dt = cut_end_set(dt, date_MDDYYYYVTTTT)  # 설연휴 마지막날
# 설연휴 첫날 2/15/2018 01:00

def tr_val_te_one(pm3, val_days):  
  tr = pm3.iloc[:-24*val_days,:]
  val = pm3.iloc[-24*val_days:-24,:] # 2주 정도 이전의 동일 요일, 시간 등 정보를 통해 검증할 수 있도록 
  te = pm3.iloc[-24:,:]
  return tr, val, te

def tr_val_te_mul(pm3, val_days, serial_num):
  # startDate: specialSerialDay first Day
  # MDDYYYYV just 1st date with 01:00
  tr = pm3.iloc[:-24*serial_num*val_days,:]
  val = pm3.iloc[-24*serial_num*val_days:-24*serial_num,:] # 2주 정도 이전의 동일 요일, 시간 등 정보를 통해 검증할 수 있도록 
  te = pm3.iloc[-24*serial_num:,:]
  return tr, val, te

def Indep17(val_days = 14):
  tr, val,te = tr_val_te_one(dt,val_days) #14일
  print("tr shape: {} val shape: {} te shape: {}".format(tr.shape, val.shape, te.shape))
  return tr, val, te

tr, val, te = Indep17()

def _deque_df(df):
    

    sequential_data = []
    prev_points = deque(maxlen = SEQ_LEN)

    for i in df.values:
        prev_points.append([n for n in i[:-1]])
        if len(prev_points) == SEQ_LEN:
            sequential_data.append([np.array(prev_points), i[-1]])

    #random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), np.array(y)



train_X, train_y = _deque_df(tr)
validation_X, validation_y = _deque_df(val)
test_X, test_y = _deque_df(te)

def to_reshape_X(train_X, validation_X, test_X):
    train_X = train_X.reshape(train_X.shape[0],train_X.shape[-1])
    validation_X = validation_X.reshape(validation_X.shape[0],validation_X.shape[-1])
    test_X = test_X.reshape(test_X.shape[0],test_X.shape[-1])
    return train_X, validation_X, test_X
train_X, validation_X, test_X = to_reshape_X(train_X, validation_X, test_X)
train_X.shape

def to_model(l1, l2):
  
  model = Sequential() # keras, default: glorot_uniform, zeros
  model.add(Dense(l1, activation='relu', input_dim = train_X.shape[1]))
  model.add(Dense(l2, activation='relu', input_dim=l1))
  model.add(Dense(1, activation='relu'))
  print(model.summary())

  #opt = tf.keras.optimizers.Adam(lr = 0.001, decay = 1e-6)
  #model.compile(loss = 'mse', optimizer = opt, metrics = ['accuracy'])
  model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['accuracy'])
  return model

model = to_model(l1,l2)

def to_fit_model(model,train_X , train_y, validation_X, validation_y, test_X, BATCH_SIZE, EPOCHS ):
  
  model.fit(train_X, train_y, batch_size = BATCH_SIZE, epochs = EPOCHS,
           validation_data = (validation_X, validation_y))
  
  pred = model.predict(test_X)
  pred = list(itertools.chain(*pred))
  return pred

pred = to_fit_model(model,train_X , train_y, validation_X, validation_y, test_X, BATCH_SIZE=64, EPOCHS=10 )

def sum_mape_mul(test_y, pred, index_n):
  # index_n: the real special DAY index
  result = pd.DataFrame({'y_true':test_y,'y_pred':pred})
  total_mape = np.mean(np.abs(test_y-pred)/test_y)*100
  test_y_24 = np.asarray(result['y_true'][index_n*24:index_n*24+24])
  # print(test_y_24)
  pred_24 = np.asarray(result['y_pred'][index_n*24:index_n*24+24])
  part_mape = np.mean(np.abs(test_y_24-pred_24)/test_y_24)*100
  print("Total MAPE: {} Day MAPE: {}".format(total_mape.round(4),part_mape.round(4)))
  return total_mape, part_mape

def sum_mape_one(test_y, pred):
  result = pd.DataFrame({'y_true':test_y,'y_pred':pred})
  total_mape = np.mean(np.abs(test_y-pred)/test_y)*100
  return total_mape

def dayMape(index_n):
  test_y_24 = np.asarray(result['y_true'][index_n*24:index_n*24+24])
  # print(test_y_24)
  pred_24 = np.asarray(result['y_pred'][index_n*24:index_n*24+24])
  return test_y_24, pred_24

total_mape = sum_mape_one(test_y, pred)

#test_y_24, pred_24 = dayMape(-3)

result = pd.DataFrame({'y_true':np.asarray(test_y)*float(100000.0),'y_pred':(np.asarray(pred)*float(100000.0))})
print(result)

plt.figure(figsize=(16,8))

plt.title(TITLE + "Day MAPE = {}".format( total_mape.round(4)))
plt.plot(result.y_pred, label='predict')
plt.plot(result.y_true, label='true', linestyle = ':')
plt.legend()
plt.show()

# plt.figure(figsize=(16,8))
# #plt.title("Childeren's Day from time01-21st to time23-30th Dec, MAPE= {}".format(total_mape.round(4)))
# plt.title(TITLE + "Day MAPE = {}".format(part_mape.round(4)))
# plt.plot(pred_24, label='predict')
# plt.plot(test_y_24, label='true', linestyle = ':')
# plt.legend()
# plt.show()

