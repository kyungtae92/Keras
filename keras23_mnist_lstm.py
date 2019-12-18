from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print(X_train[0])
print(Y_test[0])
print(X_train.shape)    # (60000, 28, 28) 행무시
print(X_test.shape)     # (10000, 28, 28) 행무시
print(Y_train.shape)
print(Y_test.shape)

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
import tensorflow as tf

X_train = X_train.reshape(X_train.shape[0], 28*28, 1).astype('float32') / 255 # (60000, 28, 28, 1) convolution layer로 바꾸기위해 와꾸 맞춤

X_test = X_test.reshape(X_test.shape[0], 28*28, 1).astype('float32') / 255 # (10000, 28, 28, 1) convolution layer로 바꾸기위해 와꾸 맞춤

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
print(Y_train[0])
print(Y_train.shape)
print(Y_test.shape)

# one hot encoding : https://wikidocs.net/22647
# 원-핫 인코딩을 두 가지 과정으로 정리해보겠습니다.
# (1) 각 단어에 고유한 인덱스를 부여합니다. (정수 인코딩)
# (2) 표현하고 싶은 단어의 인덱스의 위치에 1을 부여하고, 다른 단어의 인덱스의 위치에는 0을 부여합니다.

# 컨블루션 신경망의 설정
model = Sequential()
model.add(LSTM(32, input_shape=(28*28,1), activation='relu'))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(10))
#컬럼이10개 구조는...4-1분
# model.summary()

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])
# 분류모델에선 무조건 categorical_crossentropy 사용

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# 모델의 실행
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                    epochs=2, batch_size=200, verbose=1, # epochs=30
                    callbacks=[early_stopping_callback]) #,checkpointer])

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))
