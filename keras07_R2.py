from keras.models import Sequential
from keras.layers import Dense

import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])
# x_predict = np.array([21,22,23,24,25])

model = Sequential()
# model.add(Dense(1000, input_dim=1, activation='relu')) # dimension : 차원 / 컬럼(열)이1개이면 예측율이 떨어짐 / input_dim=1 과 input_shape=(1, ) 같은 뜻(컬럼(열)이 하나가 들어간다는 뜻)
model.add(Dense(1000, input_shape=(1, ), activation='relu')) # shape : 모양(와꾸) / 모델 짜기는 가로세로 와꾸 맞추기
model.add(Dense(60))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', 
                            # metrics=['accuracy'])
                            metrics=['mse'])
model.fit(x_train, y_train, epochs=100)

loss, mse = model.evaluate(x_test, y_test, batch_size=1)  # a[0], a[1] / evaluate를 반환하게 되면 loss, acc 를 반환
print("mse : ", mse)
print("loss : ", loss)

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

# R2가 1이면 좋고 0이면 나쁘다 정도. 정확한 것은 아님
# RMSE는 낮을수록 R2는 높을수록
#                 R2는 acc와 유사한 지표
# 선형회귀모델(y=wx+b)에서 통상적으로 지표로 가장많이 쓰는 건 RMSE와 R2
# R2를 acc로 했을 때 문제점은 acc가 1이었던 것

'''
mse :  3.6993522371631116e-05
loss :  3.6993522371631116e-05
[[11.004776]
 [12.005205]
 [13.005616]
 [14.005944]
 [15.006116]
 [16.006268]
 [17.00644 ]
 [18.006596]
 [19.00671 ]
 [20.006823]]
RMSE :  0.006082795878572825
R2 :  0.9999955151023393
'''
# R2를 높이면 RMSE는 작아짐
