from keras.models import Sequential  # keras.models 안에 Sequential를 가져옴
from keras.layers import Dense

import numpy as np # numpy를 가져오고 앞으로 numpy를 np로 줄여쓰겠단 말
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

model = Sequential()
model.add(Dense(1000, input_dim=1, activation='relu'))
model.add(Dense(900))
model.add(Dense(800))
model.add(Dense(700))
model.add(Dense(600))
model.add(Dense(500))
model.add(Dense(400))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100, batch_size=1)

mse = model.evaluate(x, y, batch_size=1)
print("mse : ", mse)