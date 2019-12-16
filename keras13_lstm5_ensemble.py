# keras13_lstm4를 카피해서 x와 y데이터를 2개로 분리하고 2개의 인풋, 2개의 아웃풋 모델인 ensemble모델을 구현하시오.

from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x1 = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
           [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12]])
x2 = array([[20,30,40], [30,40,50], [40,50,60]])
y1 = array([4,5,6,7,8,9,10,11,12,13])
y2 = array([50,60,70])
# print(x1)
# print("x1.shape :", x1.shape) # (10, 3)
# print("x2.shape :", x2.shape) # (3, 3)
# print("y1.shape :", y1.shape) # (10, )
# print("y2.shape :", y2.shape) # (3,  )

x1 = x1.reshape((x1.shape[0], x1.shape[1], 1))
x2 = x2.reshape((x2.shape[0], x2.shape[1], 1))

print(x1)
print(x2)
print("x1.shape :", x1.shape)     # (10(행무시), 3(column), 1(1은 자르는 수))
print("x2.shape :", x2.shape)     # (3(행무시), 3(column), 1(1은 자르는 수))
print("y1.shape :", y1.shape)     # (10, )
print("y2.shape :", y2.shape)     # (3, )

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=33, test_size=0.4, shuffle=False)
x1_test, x1_val, y1_test, y1_val = train_test_split(x1_test, y1_test, random_state=33, test_size=0.5, shuffle=False)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, random_state=33, test_size=0.4, shuffle=False)
x2_test, x2_val, y2_test, y2_val = train_test_split(x2_test, y2_test, random_state=33, test_size=0.5, shuffle=False)

#2. 모델 구성
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(3,1))) # input_shape = (행(무시), 3(column), 1(자르는 수))
model.add(Dense(47))
model.add(Dense(44))
model.add(Dense(33))
model.add(Dense(30))
model.add(Dense(1))

model.add(LSTM(50, activation='relu', input_shape=(3,1))) # input_shape = (행(무시), 3(column), 1(자르는 수))
model.add(Dense(47))
model.add(Dense(44))
model.add(Dense(33))
model.add(Dense(30))
model.add(Dense(1))


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

from keras.layers.merge import concatenate
merge1 = concatenate([model, model])

output1 = Dense(30)(merge1)
output1 = Dense(13)(output1)
output1 = Dense(3)(output1)

output2 = Dense(15)(merge1)
output2 = Dense(32)(output2)
output2 = Dense(3)(output2)

model = Model(inputs = [input1, input2], outputs = [output1, output2])
model.summary()
'''
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
'''