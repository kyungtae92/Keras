# OVERFITING과적합  https://tensorflow.rstudio.com/guide/keras/training_visualization/
# acc최고점, loss최저점을 어떻게 알 수 있을까?(overfit극복)
# 여러가지 있음 그 중 1가지는 callbacks이용

from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
           [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
           [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
print(x)
print("x.shape :", x.shape) # (13, 3)
print("y.shape :", y.shape) # (13,  )

x = x.reshape((x.shape[0], x.shape[1], 1))
print(x)
print("x.shape :", x.shape)     # (13(행무시), 3(column), 1(1은 자르는 수))

#2. 모델 구성
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(3,1))) # input_shape = (행(무시), 3(column), 1(자르는 수))
model.add(Dense(47))
model.add(Dense(44))
model.add(Dense(33))
model.add(Dense(30))
# model.add(Dense(25))
# model.add(Dense(20))
# model.add(Dense(17))
model.add(Dense(1))
# model.summary()

#3. 실행
model.compile(optimizer='adam', loss=['mse'])

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto') #11분
# loss값을 모니터해서 과적합이 생기면 100번 더 돌고 끊음
# mode=auto loss면 최저값이100번정도 반복되면 정지, acc면 최고값이 100번정도 반복되면 정지
# mode=min, mode=max
# early_stopping = EarlyStopping(monitor='mse', patience=10, mode='auto') 
# model.fit(x, y, epochs=330, batch_size=1, verbose=0)    # verbose=0하면 결과만 보여줌 / verbose=1 디폴트 / verbose=2 간략히나옴
model.fit(x, y, epochs=5000, callbacks=[early_stopping])




# predict 데이터
x_input = array([25,35,45]) # 1,3,?
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input)
print(yhat)

