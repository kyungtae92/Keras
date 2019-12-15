from keras.models import Sequential  # keras.models 안에 Sequential를 가져옴
from keras.layers import Dense

import numpy as np # numpy를 가져오고 앞으로 numpy를 np로 줄여쓰겠단 말 / numpy는 쉽게 list로 생각하기
x = np.array([1,2,3,4,5]) # 1행 5열
y = np.array([1,2,3,4,5]) # 1행 5열

model = Sequential() # 순차적인 모델을 만들겠다
model.add(Dense(5, input_dim=1, activation='relu')) # 순차적인 모델에다 한단을 더 쌓겠다. / input 1 output 5 / dimension : 차원
model.add(Dense(3)) # 위에 output이 아래의 input이 됨. / input 5 output 1
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam') # compile : 기계어로 변역 시키는 것(기계가 알아먹도록) / loss = cost : 손실 / optimizer : 최적화 / 나는 mse 방식의 손실함수를 사용하고 최저의 손실을 보기위해 최적값으로 adam을 사용하겠다
model.fit(x,y, epochs=100, batch_size=1) # fit : traning(기계를 훈련 시키는 것) / epochs=100 : 100번 훈련 / batch_size=1 : 1개씩 잘라서 훈련시키겠다

mse = model.evaluate(x, y, batch_size=1) # 평균 제곱 오차 mse(mean squared error) / evaluate : 평가
print("mse : ", mse)

# y에 4 -> 3.5 바꿔서 실행했을때 mse가 높게 나옴
# 즉, 데이터가 잘못됐거나 모델을 잘못만들었다는 뜻