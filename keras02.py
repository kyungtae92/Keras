from keras.models import Sequential
from keras.layers import Dense

import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
x2 = np.array([11,12,13,14,15])

model = Sequential()
model.add(Dense(1000, input_dim=1, activation='relu'))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(x,y, epochs=100) # batch_size 삭제

loss, acc = model.evaluate(x, y) # batch_size 삭제
print("acc : ", acc)
print("loss : ", loss)

y_predict = model.predict(x2)
print(y_predict)

'''
batch_size 삭제 후

acc :  1.0
loss :  0.0008015514467842877
[[10.979511]
 [11.972313]
 [12.965123]
 [13.957928]
 [14.95074 ]]
 '''
# batch_size : 작업 단위
# 값(11~15)이 엉망이 됨. 이유는? 지워도 에러나지않고 돌아간다는 건 디폴트값(32)이 있다는 말
# 10개 데이터를 32개씩 잘라서 통으로 하겠단 말이고 결국 전제적으로 세부적인 데이터를 훈련시키지 않고 통으로 하니까 제대로 안나옴
# 많은 데이터를 batch_size=1 로 하면 효율이 떨어지지만 batch_size=32로 하면 효율이 좋음
# batch_size=1 일 경우가 더 좋다면 batch_size=1 더 좋은 정답