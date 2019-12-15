#1. 데이터
import numpy as np

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])
x_val = np.array([101,102,103,104,105])
y_val = np.array([501,207,1030,1004,105])

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
# model.add(Dense(1000, input_dim=1, activation='relu')) # dimension : 차원 / 컬럼(열)이1개이면 예측율이 떨어짐 / input_dim=1 과 input_shape=(1, ) 같은 뜻(컬럼(열)이 하나가 들어간다는 뜻)
model.add(Dense(1000, input_shape=(1, ), activation='relu')) # shape : 모양(와꾸) / 모델 짜기는 가로세로 와꾸 맞추기
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

# model.summary()

#3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x_train, y_train, epochs=100, batch_size=1)
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val)) # validation : 검증(머신 자체가 평가하는 것)


#4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)  # a[0], a[1] / evaluate를 반환하게 되면 loss, acc 를 반환
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

'''
mse :  9.929223097060458e-07
[[11.000428]
 [12.000499]
 [13.000617]
 [14.000738]
 [15.000868]
 [16.00099 ]
 [17.001118]
 [18.001238]
 [19.001366]
 [20.001488]]
RMSE :  0.0009975347190324384
R2 :  0.999999879384786


y_val : 102 -> 107 변경
mse :  3.070816455874592e-05
[[10.995529]
 [11.995295]
 [12.99506 ]
 [13.994831]
 [14.994602]
 [15.994376]
 [16.994152]
 [17.993921]
 [18.993706]
 [19.993494]]
RMSE :  0.005541739769538003
R2 :  0.9999962774691306


y_val : 501,207,1030,1004,105 변경
mse :  5.954177140665706e-06
[[11.00163 ]
 [12.0018  ]
 [13.001968]
 [14.00214 ]
 [15.002307]
 [16.002476]
 [17.002645]
 [18.002817]
 [19.002983]
 [20.003157]]
RMSE :  0.002441285827021155
R2 :  0.9999992775907286
'''
# 소수 4, 5자리 바뀌는 것도 많이 바뀌는 것