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
model.fit(x,y, epochs=100, batch_size=1)

loss, acc = model.evaluate(x, y, batch_size=1)
print("acc : ", acc)
print("loss : ", loss)

y_predict = model.predict(x2)
print(y_predict)

# loss값 줄이고 싶으면 epochs을 늘리거나 노드/레이어를 늘려보기(하이퍼파라미터튜닝)
# 노드/레이어를 무작정 늘린다고 loss값이 떨어지고 acc가 올라가는건 아님 결국 최적의 값은 아무도 모름. 결국 자신이 튜닝해서 찾아야함.
'''
튜닝 전 
acc :  1.0
loss :  0.00010383744120190386
[[11.01486 ]
 [12.016556]
 [13.018249]
 [14.019781]
 [15.021117]]
튜닝 후
 acc :  1.0
loss :  1.073836731393385e-05
[[11.003096]
 [12.003911]
 [13.004727]
 [14.005544]
 [15.006357]]
 '''