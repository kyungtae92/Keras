import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1 .데이터
a = np.array(range(1, 11))

size = 5
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a, size)
print("=====================")
print(dataset)

x_train = dataset[:,0:4]
# x_train = dataset[:,0:-1]
y_train = dataset[:,4]
# y_train = dataset[:,-1]

print(x_train.shape)    # (6, 4)
print(y_train.shape)    # (6,  )

x_train = x_train.reshape((6, 4, 1))

#2. 모델
# lstm으로 수정하기
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(4,1)))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))
model.summary()

#3. 실행
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

x2 = np.array([7,8,9,10])   #(4, ) -> (1, 4)
x2 = x2.reshape((1,4,1))

#4. 예측
y_pred = model.predict(x2)
print(y_pred)
