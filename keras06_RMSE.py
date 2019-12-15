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
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', 
                            # metrics=['accuracy'])
                            metrics=['mse'])
model.fit(x_train, y_train, epochs=100)

loss, mse = model.evaluate(x_test, y_test)  # a[0], a[1] / evaluate를 반환하게 되면 loss, acc 를 반환
print("mse : ", mse)    # acc : 1.0
print("loss : ", loss)  # loss : 0.008635124191641808

'''
y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하는 수식
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): # (원래 값 ,x_test를 훈련시켜서 나온값) 둘을 비교
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
'''
# sklearn -> 머신러닝 기법
# RMSE는 mse에 루트를 씌운 것
# RMSE 낮을수록 좋음
# metrics : 출력할 때 나한테 보여지는 공간
# metrics=['mse'] 일땐 loss와 mse가 같게 나옴
# 딥러닝에선 딱 두가지 회귀와 분류가 있음
# 분류모델에서 사용하는 것이 accuracy
# 회귀모델(소수점이 나타나는 모델) accuracy 사용X