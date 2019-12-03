from keras.models import Sequential
from keras.layers import Dense

import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])
# x_predict = np.array([21,22,23,24,25])

model = Sequential()
# model.add(Dense(1000, input_dim=1, activation='relu'))
model.add(Dense(1000, input_shape=(1, ), activation='relu'))
# model.add(Dense(500))
# model.add(Dense(300))
# model.add(Dense(200))
# model.add(Dense(100))
# model.add(Dense(90))
# model.add(Dense(80))
# model.add(Dense(60))
# model.add(Dense(50))
# model.add(Dense(30))
# model.add(Dense(10))

model.add(Dense(900))
model.add(Dense(700))
model.add(Dense(600))
model.add(Dense(500))
model.add(Dense(400))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam',
            #   metrics=['accuracy'])
            metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

loss, mse = model.evaluate(x_test, y_test, batch_size=1)  # a[0], a[1]
print("mse : ", mse)    # 1.0
print("loss : ", loss)  # 0.008138901554048061

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

# 문제 1. R2를 0.5 이하로 줄이시오.
# 레이어는 input과 output 포함 5개 이상, 노드는 각 레이어당 5개 이상
# batch_size = 1
# epochs = 100 이상
'''
mse :  6.321906089782715
loss :  6.3219057559967045
[[12.746864]
 [13.907876]
 [15.068893]
 [16.229916]
 [17.39093 ]
 [18.551947]
 [19.712957]
 [20.873976]
 [22.034998]
 [23.196014]]
RMSE :  2.514337750929305
R2 :  0.23370977869718357
'''