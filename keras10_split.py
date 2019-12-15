#1. 데이터
import numpy as np

x = np.array(range(1, 101)) # 1~100
y = np.array(range(1, 101))
print(x)
# 6 : 2 : 2
# x_train = x[:60] # 1~60
# x_val = x[60:80] # 61~80
# x_test = x[80:] # 81~100
# y_train = y[:60]
# y_val = y[60:80]
# y_test = y[80:]
# print(x_train)
# print(x_val)
# print(x_test)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.4, shuffle=False) # 6:4 / 섞기 싫으면 shuffle=False / default=True
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, random_state=66, test_size=0.5, shuffle=False) # 6:2:2
print(x_train)
print(x_test)
print(x_val)

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
# model.add(Dense(1000, input_dim=1, activation='relu')) # dimension : 차원 / 컬럼(열)이1개이면 예측율이 떨어짐 / input_dim=1 과 input_shape=(1, ) 같은 뜻(컬럼(열)이 하나가 들어간다는 뜻)
model.add(Dense(1000, input_shape=(1, ), activation='relu')) # shape : 모양(와꾸) / 모델 짜기는 가로세로 와꾸 맞추기
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

# model.summary()

#3. 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# model.fit(x_train, y_train, epochs=100, batch_size=1)
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val)) # validation은 검증(머신 자체가 평가하는 것)


#4. 평가 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)  # a[0], a[1] / evaluate를 반환하게 되면 loss, acc 를 반환
print("mse : ", mse)

y_predict = model.predict(x_test)
print(y_predict)

# RMSE 구하는 수식
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)

# train_test_split 사용법
# http://blog.naver.com/PostView.nhn?blogId=siniphia&logNo=221396370872&from=search&redirect=Log&widgetTypeCall=true&directAccess=false
'''
[  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36
  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54
  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72
  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90
  91  92  93  94  95  96  97  98  99 100]

x_train
[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48
 49 50 51 52 53 54 55 56 57 58 59 60]

x_test
[61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80]

x_val
[ 81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98
  99 100]

<test_size>
RMSE :  0.019709663009811147
R2 :  0.9999883166671891

<train_size>
RMSE :  1.0862652872548673
R2 :  0.9842495358269869

<test_size 와 train_size 둘다 적용했을 때>
train이 적용됨
'''