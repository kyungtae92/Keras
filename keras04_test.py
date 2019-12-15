from keras.models import Sequential
from keras.layers import Dense

import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])

model = Sequential()
model.add(Dense(100, input_dim=1, activation='relu'))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=300, batch_size=1)

loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("acc : ", acc)
print("loss : ", loss)

y_predict = model.predict(x_test)
print(y_predict)

'''
acc :  1.0
loss :  9.404175216332078e-11
[[10.999983]
 [11.999985]
 [12.999989]
 [13.999991]
 [14.999997]
 [16.      ]
 [17.000004]
 [18.000008]
 [19.00001 ]
 [20.000015]]
 '''
# keras03_summary.py에서는 결과를 훈련데이터로 측정했기때문에 당연히 acc 100나옴
# 훈련 시켰으면 훈련에 대한 평가는 다른 데이터로 해야함(머신이 답을 외우는 걸 방지하기위해)
# 통장적으로 x_predic 와 x_test를 유사하게 사용하거나 아예 새로운 것을 사용하거나
# 통상적으로 train 과 test 의 비율은 7:3