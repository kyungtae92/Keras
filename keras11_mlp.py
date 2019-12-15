#1. 데이터
import numpy as np

x = np.array([range(1, 101), range(101, 201)]) # 데이터 2개(2개 이상이면 리스트로 묶기)
y = np.array([range(1, 101), range(101, 201)])
print(x)

print(x.shape) # (2, 100)

x = np.transpose(x) # transpose : 열과 행을 바꿈
y = np.transpose(y)

print(x.shape) # (100, 2)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.4, shuffle=False) # 6:4 / 섞기 싫으면 shuffle=False / default=True
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, random_state=66, test_size=0.5, shuffle=False) # 6:2:2

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

# model.add(Dense(1000, input_dim=2, activation='relu'))
model.add(Dense(100, input_shape=(2, ), activation='relu')) # column이 2개 다른말로 input_dim=2 / shape=(1, ) -> shape(?, 1) : 행은 필요 없고 열을 1개쓰겠다. 벡터가 1개라는 것.
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(2)) # input colomn이 2개이므로 output도 2개

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

# 행무시!!! 데이터가 계속 추가 될 수 있으므로 열만 들어감