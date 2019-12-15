#1. 데이터
import numpy as np

x = np.array(range(1, 101)) # 1~100
y = np.array(range(1, 101))
print(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.4, shuffle=False) # 6:4 / 섞기 싫으면 shuffle=False / default=True
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, random_state=66, test_size=0.5, shuffle=False) # 6:2:2

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

# input1 = Input(shape=(1,)) # 최초의 Input 명시, 컬럼(열) 1개
# dense1 = Dense(5, activation='relu')(input1)
# dense2 = Dense(3)(dense1) # input 5, output 3
# dense3 = Dense(4)(dense2) # input 3, output 4
# dense4 = Dense(4)(dense3)
# dense5 = Dense(4)(dense4)
# dense6 = Dense(4)(dense5)
# dense7 = Dense(4)(dense6)
# dense8 = Dense(4)(dense7)
# dense9 = Dense(4)(dense8)
# dense10 = Dense(4)(dense9)
# output1 = Dense(1)(dense10)

input1 = Input(shape=(1,))
xx = Dense(5, activation='relu')(input1)
xx = Dense(3)(xx)
xx = Dense(4)(xx)
xx = Dense(4)(xx)
xx = Dense(4)(xx)
xx = Dense(4)(xx)
xx = Dense(4)(xx)
xx = Dense(4)(xx)
xx = Dense(4)(xx)
xx = Dense(4)(xx)
output1 = Dense(1)(xx)

model = Model(inputs = input1, outputs = output1) # 이 Model은 inpt1~output1이다라고 명시해줘야 함
model.summary()
'''
#3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x_train, y_train, epochs=100, batch_size=1)
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val)) # validation은 검증(머신 자체가 평가하는 것)

#4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)  # a[0], a[1]
print("mse : ", mse)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하는 수식
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

# 레이어를 10개 이상 늘리시오.
'''
'''
mse :  8.221832242805149e-11
[[61.00002 ]
 [62.      ]
 [63.000004]
 [64.00001 ]
 [65.000015]
 [66.000015]
 [67.00001 ]
 [68.00003 ]
 [69.00001 ]
 [70.      ]
 [71.00001 ]
 [72.000015]
 [72.99999 ]
 [74.00002 ]
 [74.99999 ]
 [76.00003 ]
 [77.000015]
 [77.99999 ]
 [79.      ]
 [80.00001 ]]
RMSE :  1.4425407715750914e-05
R2 :  0.9999999999937416
'''