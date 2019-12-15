# 문제 1. R2를 0.5 이하로 줄이시오.
# 레이어는 input과 output 포함 5개 이상, 노드는 각 레이어당 5개 이상
# batch_size = 1
# epochs = 100 이상

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
model.add(Dense(80))
model.add(Dense(600))
model.add(Dense(500))
model.add(Dense(60))
model.add(Dense(850))
model.add(Dense(705))
model.add(Dense(400))
model.add(Dense(605))
model.add(Dense(500))
model.add(Dense(305))
model.add(Dense(100))
model.add(Dense(450))
model.add(Dense(520))
model.add(Dense(501))
model.add(Dense(509))
model.add(Dense(301))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', 
                            # metrics=['accuracy'])
                            metrics=['mse'])
model.fit(x_train, y_train, epochs=500, batch_size=1)

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

from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)


'''
mse :  398596.34375
loss :  398596.33525390626
[[  -80.73349]
 [ -110.66904]
 [ -195.94003]
 [ -284.6295 ]
 [ -406.70517]
 [ -534.7559 ]
 [ -657.33014]
 [ -802.9805 ]
 [ -953.9854 ]
 [-1108.3292 ]]
RMSE :  631.349600777605
R2 :  -48314.43253358075
'''
# 결국 많은 훈련(과부화)을 하게 되면 R2 값은 낮아짐