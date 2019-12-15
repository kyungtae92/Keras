#1. 데이터
import numpy as np

x = np.array([range(1, 101), range(101, 201)]) # 데이터 2개(2개 이상이면 리스트로 묶기)
y = np.array([range(201, 301)])
print(x)

print(x.shape) # (2, 100)
print(y.shape) # (1, 100)

x = np.transpose(x)
y = np.transpose(y)

print(x.shape) # (100, 2)
print(y.shape) # (100, 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.4, shuffle=False) # 6:4 / 60:2 40:2 60:1 40:1
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, random_state=66, test_size=0.5, shuffle=False) # 6:2:2 / 20:2 20:2 20:1 20:1
# print(x_train)
# print(x_test)
# print(x_val)

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# model.add(Dense(100, input_dim=2, activation='relu'))
model.add(Dense(50, input_shape=(2, ), activation='relu')) # column이 2개 다른말로 input_dim=2 / input_dim=1 이면 shape=(1, ) 기억 
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

# model.summary()

#3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x_train, y_train, epochs=100, batch_size=1)
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val)) # validation은 검증(머신 자체가 평가하는 것)


#4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)  # a[0], a[1] / evaluate를 반환하게 되면 loss, acc 를 반환
print("mse : ", mse)

# aaa = np.array([[101,102,103],[201,202,203]]) # (2, 3) / 대괄호 주의
# aaa = np.transpose(aaa) # (3, 2)
# y_predict = model.predict(aaa)

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

'''
<column 2개를 집어넣어서 column 1개가 나오개하는 것>

input이 2개이고 input과 전혀다르고, 1개 라면?
ex) x1 삼성전자 주가 x2 sk주가 y1 종합 주가
삼성, sk주가를 넣고 내일의 종합주가를 예측.?
x = np.array([range(1,101), range(101,201)])
y = np.array([range(201,301)]) 
x는 (100,2)   y는 (100,1) 이어도
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.4)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, random_state=66, test_size=0.5)
x_train (60,2) x_test (40,2) y_train (60,1) y_test (40,1)
x_val (20,2) x_test (20,2) y_val (20,1) y_test (20,1)   
마지막 아웃풋 모델만 model.add(Dense(1))로 수정. 다른것 수정 필요 없음


<predict를 x_test 말고 다른 값 예측해보기>
aaa = np.array([[101,102,103],[201,202,203]]) # (2, 3)
aaa = np.transpose(aaa) # (3, 2)
y_predict = model.predict(aaa)
print(y_predict)

mse :  0.005959832109510899
[[301.5782 ]
 [302.59717]
 [303.6161 ]]


 mse :  0.015868453308939934
[[261.01547]
 [262.00296]
 [262.9905 ]
 [263.97797]
 [264.96548]
 [265.95297]
 [266.94043]
 [267.92798]
 [268.91547]
 [269.90295]
 [270.89047]
 [271.87796]
 [272.86545]
 [273.85297]
 [274.84042]
 [275.8279 ]
 [276.81543]
 [277.80295]
 [278.79044]
 [279.77792]]
RMSE :  0.12596611766985477
R2 :  0.9995227830736597
'''

