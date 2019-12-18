# Normalization(일반화) -레이어의 가중치(w)를 표준화 하는 것(레이어의 가중치를 모아주는 것)
# 정규화, 표준화는 데이터단계
# 일반화는 레이어단계 
# BatchNormalization과 Dropout 같이 써도 에러는 나지 않지만 같이쓰면 과적합 회피가 잘되지는 않음 통상적으로 둘 중 하나만 사용
# GAN이라는 기법에서는 BatchNormalization과 Dropout 같이 사용

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
model.add(Dense(1000, input_shape=(1, ), activation='relu')) # shape : 모양(와꾸) / 모델 짜기는 가로세로 와꾸 맞추기
model.add(Dense(1000))
model.add(BatchNormalization()) # 두번째 레이어의 가중치를 모아주는 것
model.add(Dropout(0.2))
model.add(Dense(1000))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1000))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1000))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1000))
model.add(Dense(1))

# model.summary()

#3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x_train, y_train, epochs=100, batch_size=1)
model.fit(x_train, y_train, epochs=10, batch_size=1, validation_data=(x_val, y_val)) # validation은 검증(머신 자체가 평가하는 것)


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