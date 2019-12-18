#1. 데이터
import numpy as np

x = np.array(range(1, 101)) # 1~100
y = np.array(range(1, 101))
print(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.4, shuffle=False) # 6:4 / 섞기 싫으면 shuffle=False / default=True
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, random_state=66, test_size=0.5, shuffle=False) # 6:2:2
print(x_train)
print(x_test)
print(x_val)

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization

model = Sequential()
# model.add(Dense(1000, input_dim=1, activation='relu')) # dimension : 차원 / 컬럼(열)이1개이면 예측율이 떨어짐 / input_dim=1 과 input_shape=(1, ) 같은 뜻(컬럼(열)이 하나가 들어간다는 뜻)
model.add(Dense(5, input_shape=(1, ), activation='relu'))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

# model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

import keras
tb_hist = keras.callbacks.TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
from keras.callbacks import EarlyStopping, TensorBoard
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')

model.fit(x_train, y_train, epochs=10, batch_size=1, validation_data=(x_val, y_val), callbacks=[early_stopping, tb_hist])

#4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
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