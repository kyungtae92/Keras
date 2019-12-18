# 모델이 잘 인식할 수 있게 데이터를 전처리 해주어야 함
# 소수의 데이터가 값이 크다. 예측값을 100이나 1000으로 주어져도 10000단위로 예측 될 가능성이 많다.
# 정규화 방법1 - MinMax Scaler 1~60000의 범위를 0~1사이로
# 정규화 공식 : x-최소/최대-최소

from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터  # 문제점 : 이 데이터의 구조가 회귀 모델에 사용하기 적합하지 않음
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
           [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
           [20000,30000,40000], [30000,40000,50000],
           [40000,50000,60000]])
y = array([4,5,6,7,8,9,10,11,12,13,50000,60000,70000])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x) # 실행
x = scaler.transform(x) # MinMaxScaler 사용할 땐 fit한 다음 transform 해주기! evaluate, predict과정과 같음
print(x)
'''
0~1사이로 압축
[[0.00000000e+00 0.00000000e+00 0.00000000e+00] -> [[1,2,3]
 [2.50006250e-05 2.00008000e-05 1.66675000e-05]     .
 [5.00012500e-05 4.00016001e-05 3.33350001e-05]     .
 [7.50018750e-05 6.00024001e-05 5.00025001e-05]     .
 [1.00002500e-04 8.00032001e-05 6.66700002e-05]     .
 [1.25003125e-04 1.00004000e-04 8.33375002e-05]     .
 [1.50003750e-04 1.20004800e-04 1.00005000e-04]     .
 [1.75004375e-04 1.40005600e-04 1.16672500e-04]     .
 [2.00005000e-04 1.60006400e-04 1.33340000e-04]     .
 [2.25005625e-04 1.80007200e-04 1.50007500e-04]     .
 [4.99987500e-01 5.99983999e-01 6.66649999e-01]     .
 [7.49993750e-01 7.99992000e-01 8.33325000e-01]     .
 [1.00000000e+00 1.00000000e+00 1.00000000e+00] -> [40000,50000,60000]]
'''
# 머신이 계산이 빨라지고 예측률도 올라가지만 아직 데이터가 치우쳐있다는 문제점이 있음

print(x)
print("x.shape :", x.shape) # (13, 3)
print("y.shape :", y.shape) # (13,  )

#2. 모델 구성
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(3, )))
model.add(Dense(40)) # activation='linear'는 default
model.add(Dense(33))
model.add(Dense(20))
model.add(Dense(1))
# model.summary()

#3. 실행
model.compile(optimizer='adam', loss=['mse'])
model.fit(x, y, epochs=100, batch_size=1, verbose=1)

# 평가 예측
x_input = array([25,35,45]) # (3, )
x_input = x_input.reshape((1,3))
x_input = scaler.transform(x_input)
# x, y에서 사용한 scaler 가중치 그대로 사용해야한다. x_input의 범위는 25-45

yhat = model.predict(x_input, verbose=1)
print(yhat)

# x값은 전처리가 되어있지만 x_input값은 전처리가 되어있지않아서 값이 제대로 나오지 않음
# 데이터는 나중에 전처리하면 매핑값과에서 문제가 생길 수 있으므로 초반에 전처리하고 작업하고 y는 전처리 할 필요가 없다(x값을 전처리해도 y값과 매치되는 값은 변함이 없으므로 y값 전처리 해줄 필요 없음)
# 데이터 스케일러 https://mkjjo.github.io/python/2019/01/10/scaler.html