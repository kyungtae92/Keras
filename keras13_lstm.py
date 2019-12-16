from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = array([4,5,6,7])
print(x)
# 데이터 잡으면 먼저 shape하기
print("x.shape :", x.shape)     # (4, 3)
print("y.shape :", y.shape)     # (4,  ) <- 벡터가 4개
'''
데이터 구조
 x      y
123     4
234     5
345     6
456     7
'''
x = x.reshape((x.shape[0], x.shape[1], 1))
print(x)
print("x.shape :", x.shape)     # (4(행무시), 3(column), 1(1은 자르는 수)) / [1],[2],[3]

#2. 모델 구성
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(3,1))) # input_shape = (행(무시), 3(column), 1(자르는 수))
model.add(Dense(47))
model.add(Dense(44))
model.add(Dense(33))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(20))
model.add(Dense(17))
model.add(Dense(1))
# model.summary()

#3. 실행
model.compile(optimizer='adam', loss=['mse'])
model.fit(x, y, epochs=550, batch_size=1)

x_input = array([6,7,8]) # 1,3,?
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input)
print(yhat)