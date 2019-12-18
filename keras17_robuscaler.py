# 잘 사용하지는 않음
# RobustScaler(X): 중앙값(median)이 0, IQR(interquartile range)이 1이 되도록 변환.
# MaxAbsScaler(X): 0을 기준으로 절대값이 가장 큰 수가 1또는 -1이 되도록 변환

from numpy import array, transpose
from keras.models import Sequential
from keras.layers import Dense, LSTM

x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20000,30000,40000], [30000,40000,50000], 
            [40000,50000,60000], [100,200,300]])
y = array([4,5,6,7,8,9,10,11,12,13,50000,60000,70000,400])

print(x.shape) # (14,3)
print(y.shape) # (14,)

from sklearn.preprocessing import RobustScaler, MaxAbsScaler
scaler = RobustScaler()
scaler.fit(x)
x = scaler.transform(x)

# train : predict = 13 : 1
x_train = x[:13]
x_predict = x[13:]
y_train = y[:13]

model = Sequential()
model.add(Dense(100, activation = 'relu', input_shape=(3, )))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=1)

yhat = model.predict(x_predict)
print(yhat)