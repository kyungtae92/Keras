#1. 데이터
import numpy as np

x = np.array(range(1, 101)) # 1~100
y = np.array(range(1, 101))
print(x)
# 6 : 2 : 2
x_train = x[:60] # 1~60
x_val = x[60:80] # 61~80
x_test = x[80:] # 81~100
y_train = y[:60]
y_val = y[60:80]
y_test = y[80:]
print(x_train)
print(x_val)
print(x_test)
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
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val)) # validation은 검증(머신 자체가 평가하는 것)


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
mse :  1.2999927093915176e-05
[[ 81.00112 ]
 [ 82.00136 ]
 [ 83.001595]
 [ 84.00185 ]
 [ 85.00206 ]
 [ 86.00231 ]
 [ 87.00255 ]
 [ 88.00279 ]
 [ 89.00301 ]
 [ 90.003235]
 [ 91.00349 ]
 [ 92.00372 ]
 [ 93.003944]
 [ 94.00418 ]
 [ 95.004425]
 [ 96.004616]
 [ 97.00485 ]
 [ 98.0051  ]
 [ 99.0053  ]
 [100.005516]]
RMSE :  0.003608891117313653
R2 :  0.9999996082978918
'''
