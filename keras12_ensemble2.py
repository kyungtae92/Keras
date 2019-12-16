# Input 2 , Output 3
#1. 데이터
import numpy as np

x1 = np.array([range(100), range(311,411), range(100)])  # (3, 100)
x2 = np.array([range(501,601), range(711,811), range(100)]) # (3, 100)

y1 = np.array([range(100,200), range(311,411), range(100,200)])
y2 = np.array([range(501,601), range(711,811), range(100)])
y3 = np.array([range(401,501), range(211,311), range(100)])

x1 = np.transpose(x1)
y1 = np.transpose(y1)
x2 = np.transpose(x2)
y2 = np.transpose(y2)
y3 = np.transpose(y3)

print(x1.shape) # (100, 3)
print(x2.shape) # (100, 3)
print(y1.shape) # (100, 3)
print(y2.shape) # (100, 3) 
print(y3.shape) # (100, 3) 

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=33, test_size=0.4, shuffle=False)
x1_test, x1_val, y1_test, y1_val = train_test_split(x1_test, y1_test, random_state=33, test_size=0.5, shuffle=False)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, random_state=33, test_size=0.4, shuffle=False)
x2_test, x2_val, y2_test, y2_val = train_test_split(x2_test, y2_test, random_state=33, test_size=0.5, shuffle=False)
y3_train, y3_test = train_test_split(y3, random_state=33, test_size=0.4, shuffle=False)
y3_test, y3_val = train_test_split(y3_test, random_state=33, test_size=0.5, shuffle=False)

print(y3_test.shape)


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

output2 = Dense(15)(merge1)
output2 = Dense(32)(output2)
output2 = Dense(3)(output2)

output3 = Dense(15)(merge1)
output3 = Dense(32)(output3)
output3 = Dense(3)(output3)

model = Model(inputs = [input1, input2], outputs = [output1, output2, output3]) # 입력이 두개 이상인 것 리스트로 꼭 묶어주기
model.summary()

#3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x_train, y_train, epochs=100, batch_size=1)
model.fit([x1_train, x2_train], [y1_train, y2_train, y3_train], epochs=100, batch_size=1, validation_data=([x1_val, x2_val], [y1_val, y2_val, y3_val ])) # validation은 검증(머신 자체가 평가하는 것)


#4. 평가 예측
mse = model.evaluate([x1_test, x2_test], [y1_test, y2_test, y3_test], batch_size=1)  # a[0], a[1]
print("mse : ", mse[0])
print("mse : ", mse[1])
print("mse : ", mse[2])
print("mse : ", mse[3])
print("mse : ", mse[4])
# print("mse : ", mse[5])

y1_predict, y2_predict, y3_predict = model.predict([x1_test, x2_test])
print(y1_predict, y2_predict, y3_predict)


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
RMSE2 = RMSE(y2_test, y2_predict)
RMSE3 = RMSE(y3_test, y3_predict)
print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE3 : ", RMSE3)
print("RMSE : ", (RMSE1 + RMSE2 + RMSE3)/2)

# R2 구하기
from sklearn.metrics import r2_score
r2_y1_predict = r2_score(y1_test, y1_predict)
r2_y2_predict = r2_score(y2_test, y2_predict)
r2_y3_predict = r2_score(y3_test, y3_predict)

print("R2_1 : ", r2_y1_predict)
print("R2_2 : ", r2_y2_predict)
print("R2_3 : ", r2_y3_predict)
print("R2 : ", (r2_y1_predict + r2_y2_predict + r2_y3_predict)/2)