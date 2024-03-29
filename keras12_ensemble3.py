# Input 2 , Output 1
#1. 데이터
import numpy as np

x1 = np.array([range(100), range(311,411), range(100)])  # (3, 100)
x2 = np.array([range(501,601), range(711,811), range(100)])

y1 = np.array([range(100,200), range(311,411), range(100,200)])

x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)

print(x1.shape) # (100, 3)
print(x2.shape) # (100, 3)
print(y1.shape) # (100, 3)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=33, test_size=0.4, shuffle=False)
x1_test, x1_val, y1_test, y1_val = train_test_split(x1_test, y1_test, random_state=33, test_size=0.5, shuffle=False)
x2_train, x2_test = train_test_split(x2, random_state=33, test_size=0.4, shuffle=False)
x2_test, x2_val = train_test_split(x2_test, random_state=33, test_size=0.5, shuffle=False)

print(x2_test.shape)

#2. 모델구성(2개)
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()

input1 = Input(shape=(3,))
dense1 = Dense(5, activation='relu')(input1)
dense2 = Dense(3)(dense1)
dense3 = Dense(4)(dense2)
middle1 = Dense(3)(dense3)

input2 = Input(shape=(3,)) 
dense1 = Dense(5, activation='relu')(input2) 
dense2 = Dense(3)(dense1) 
dense3 = Dense(4)(dense2)
middle2 = Dense(3)(dense3)

# concatenate_1 (Concatenate)  (None, 6)  6인 이유는 아웃풋이 3, 3 이기때문에
# concatenate 모델 합치기
from keras.layers.merge import concatenate
merge1 = concatenate([middle1, middle2]) # 두 모델의 가장 끝 레이어의 이름을 concatenate 안에 명시 -> merge1 레이어 생성

output1 = Dense(30)(merge1)
output1 = Dense(13)(output1)
output1 = Dense(3)(output1)

model = Model(inputs = [input1, input2], outputs = output1) # 입력이 두개 이상인 것 리스트로 꼭 묶어주기
model.summary()

#3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x_train, y_train, epochs=100, batch_size=1)
model.fit([x1_train, x2_train], y1_train, epochs=100, batch_size=1, validation_data=([x1_val, x2_val], y1_val)) # validation은 검증(머신 자체가 평가하는 것)


#4. 평가 예측
mse = model.evaluate([x1_test, x2_test], y1_test, batch_size=1)  # a[0], a[1]
print("mse : ", mse)

y1_predict = model.predict([x1_test, x2_test])
print("predict : ", y1_predict)


# # RMSE 구하는 수식
# from sklearn.metrics import mean_squared_error
# def RMSE(y1_test, y1_predict):
#     return np.sqrt(mean_squared_error(y1_test, y1_predict))
# print("RMSE : ", RMSE(y1_test, y1_predict))

# def RMSE2(y2_test, y2_predict):
#     return np.sqrt(mean_squared_error(y2_test, y2_predict))
# print("RMSE2 : ", RMSE2(y2_test, y2_predict))

# # R2 구하기
# from sklearn.metrics import r2_score
# r2_y_predict = r2_score(y1_test, y1_predict)
# print("R2_1 : ", r2_y_predict)

# r2_y_predict = r2_score(y2_test, y2_predict)
# print("R2_2 : ", r2_y_predict)

# RMSE 구하는 수식
from sklearn.metrics import mean_squared_error
def RMSE(xxx, yyy):
    return np.sqrt(mean_squared_error(xxx, yyy))
RMSE1 = RMSE(y1_test, y1_predict)
print("RMSE1 : ", RMSE1)

# R2 구하기
from sklearn.metrics import r2_score
r2_y1_predict = r2_score(y1_test, y1_predict)

print("R2_1 : ", r2_y1_predict)