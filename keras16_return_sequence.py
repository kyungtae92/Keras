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
model.add(LSTM(10, activation='relu', input_shape=(3,1), return_sequences=True)) # input_shape = (행(무시), 3(column), 1(자르는 수))
model.add(LSTM(10, activation='relu', return_sequences=True)) # input_shape = (행(무시), 3(column), 1(자르는 수)) # (None, 3, 10)
model.add(LSTM(10, activation='relu', return_sequences=True)) # input_shape = (행(무시), 3(column), 1(자르는 수))
model.add(LSTM(10, activation='relu', return_sequences=True)) # input_shape = (행(무시), 3(column), 1(자르는 수))
model.add(LSTM(10, activation='relu', return_sequences=True)) # input_shape = (행(무시), 3(column), 1(자르는 수))
model.add(LSTM(10, activation='relu', return_sequences=True)) # input_shape = (행(무시), 3(column), 1(자르는 수))
model.add(LSTM(10, activation='relu', return_sequences=True)) # input_shape = (행(무시), 3(column), 1(자르는 수))
model.add(LSTM(10, activation='relu', return_sequences=True)) # input_shape = (행(무시), 3(column), 1(자르는 수))
model.add(LSTM(10, activation='relu', return_sequences=True)) # input_shape = (행(무시), 3(column), 1(자르는 수))
model.add(LSTM(10, activation='relu', return_sequences=True)) # input_shape = (행(무시), 3(column), 1(자르는 수))
model.add(LSTM(3))
model.add(Dense(5))
model.add(Dense(1))
model.summary()

#3. 실행
model.compile(optimizer='adam', loss=['mse'])

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto') #11분
# early_stopping = EarlyStopping(monitor='mse', patience=10, mode='auto') 
# model.fit(x, y, epochs=330, batch_size=1, verbose=0)    # verbose=0하면 결과만 보여줌 / verbose=1 디폴트 / verbose=2 간략히나옴
model.fit(x, y, epochs=5000, callbacks=[early_stopping])

# predict 데이터
x_input = array([25,35,45]) # 1,3,?
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input)
print(yhat)
