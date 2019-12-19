# keras13_lstm4를 카피해서 x와 y데이터를 2개로 분리하고 2개의 인풋, 2개의 아웃풋 모델인 ensemble모델을 구현하시오.

from numpy import array
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
           [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
           [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print("x.shape : ", x.shape)    # 13,3
print("y.shape : ", y.shape)    # 13,

x1 = x[0:10]
x2 = x[10:]
y1 = y[0:10]
y2 = y[10:]

print("x1.shape :", x1.shape) # (10, 3)
print("x2.shape :", x2.shape) # (3, 3)
print("y1.shape :", y1.shape) # (10, )
print("y2.shape :", y2.shape) # (3, )

x1 = x1.reshape((x1.shape[0], x1.shape[1], 1))
x2 = x2.reshape((x2.shape[0], x2.shape[1], 1))
print(x1.shape)
print(x2.shape)
print(x)

#2. 모델 구성
input1 = Input(shape=(3, 1))
xx = LSTM(40, activation = 'relu')(input1) # 3,1에서 1은 잘라서 작업할 개수
xx = Dense(30)(xx)
xx = Dense(20)(xx)
middle1 = Dense(1)(xx)

input2 = Input(shape=(3, 1))
yy = LSTM(40, activation = 'relu')(input2) # 3,1에서 1은 잘라서 작업할 개수
yy = Dense(30)(yy)
yy = Dense(20)(yy)
middle2 = Dense(1)(yy)
# model.summary()

from keras.layers.merge import concatenate, Concatenate
# merge1 = concatenate([middle1, middle2])
merge1 = Concatenate()([middle1, middle2]) # 두 모델의 가장 끝 레이어의 이름을 concatenate 안에 명시 -> merge1 레이어 생성

output1 = Dense(30)(merge1)
output1 = Dense(13)(output1)
output1 = Dense(1)(output1)

output2 = Dense(30)(merge1)
output2 = Dense(13)(output2)
output2 = Dense(1)(output2)

model = Model(inputs = [input1, input2], outputs = [output1, output2]) # 입력이 두개 이상인 것 리스트로 꼭 묶어주기
model.summary()

#3. 실행
model.compile(loss='mse', optimizer='adam', metrics=['mse'] )

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto') #11분
model.fit([x1, x1], [y1, y1], epochs=10, verbose=2, callbacks=[early_stopping])

# predict 데이터
x1_input = array([11,12,13]) # 1,3,?
x2_input = array([25,35,45]) # 1,3,?
x1_input = x1_input.reshape((1,3,1))
x2_input = x2_input.reshape((1,3,1))


yhat1, yhat2 = model.predict([x1_input, x2_input])
print(yhat1, yhat2)
