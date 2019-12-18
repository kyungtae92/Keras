from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# --------------까지 하고 실행, MNIST데이터 다운

print(X_train[0])
print(Y_test[0])
print(X_train.shape)    # (60000, 28, 28) 행무시
print(X_test.shape)     # (10000, 28, 28) 행무시    
print(Y_train.shape)    # (60000,)
print(Y_test.shape)     # (10000,)

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
import tensorflow as tf

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255 # (60000, 28, 28, 1) convolution layer로 바꾸기위해 와꾸 맞춤
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255 # (10000, 28, 28, 1) convolution layer로 바꾸기위해 와꾸 맞춤
# .astype('float32') / 255 -> minmaxscaler 이 이미지의 범위 0~255 성능^ (안했을때 acc)
Y_train = np_utils.to_categorical(Y_train) # 분류를 편하게 하기 위해
Y_test = np_utils.to_categorical(Y_test)
print(Y_train[0])   # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.] 5를 원핫인코딩
print(Y_train.shape)    # (60000, 10)
print(Y_test.shape)     # (10000, 10)

# one hot encoding : https://wikidocs.net/22647
# 원-핫 인코딩을 두 가지 과정으로 정리해보겠습니다.
# (1) 각 단어에 고유한 인덱스를 부여합니다. (정수 인코딩)
# (2) 표현하고 싶은 단어의 인덱스의 위치에 1을 부여하고, 다른 단어의 인덱스의 위치에는 0을 부여합니다.

# 컨블루션 신경망의 설정
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=2)) # (24,24,64) -> (12,12,64)
model.add(Dropout(0.25)) # 사용을 하지 않을 뿐이지 빠지는 것은 아님
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
#컬럼이10개 구조는...4-1분
# MaxPooling 가장 영향이 큰 픽셀만 뽑기. 작업시간도 줄고 결과가 전체로 했을 때와 비슷하거나 더 좋다.
# (2,2)에서 가장 큰 것 뽑음. 4*4 -> 2*2
# model.summary()

model.compile(loss='categorical_crossentropy',
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
