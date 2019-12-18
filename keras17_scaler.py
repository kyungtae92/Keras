'''
StandardScalar(표준화) - 평균이 0과 표준편차가 1이 되도록 변환. 데이터가 치우쳐져 있을 경우
표준화 공식 x - 평균/표준편차
        1 2 3 4 5
평균        3
편차    -2 -1 0 1 2  
분산    4+1+0+1+4=10/5(개수)=2
표준편차    루트2
넓게 분포된 데이터들 보다 모여진 데이터가 더 예측율이 좋음
'''
from numpy import array, transpose
from keras.models import Sequential
from keras.layers import Dense, LSTM

x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20000,30000,40000], [30000,40000,50000], 
            [40000,50000,60000], [100,200,300]])
y = array([4,5,6,7,8,9,10,11,12,13,50000,60000,70000,400])
print(x)
print(x.shape) # (14, 3)
print(y.shape) # (14,  )

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

# train : predict = 13 : 1
x_train = x[:13]
x_predict = x[13:]
y_train = y[:13]

#2. 모델 구성
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(3, )))
model.add(Dense(47))
model.add(Dense(44))
model.add(Dense(33))
model.add(Dense(30))
model.add(Dense(1))
# model.summary()

#3. 실행
model.compile(optimizer='adam', loss=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=1)

# x값 중 1개 잘라서 predict로 사용
yhat = model.predict(x_predict)
print(yhat)

# 데이터가 들쑥 날쑥이라 결과 값이 좋지 않음.
# x값은 전처리가 되어있지만 x_input값은 전처리가 되어있지않아서 값이 제대로 나오지 않음