# x_train 60000, 28, 28 -> x1, x2, 각30000
# y도 동일
from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print(X_train[0])
print(Y_test[0])
print(X_train.shape)    # (60000, 28, 28) 행무시
print(X_test.shape)     
print(Y_train.shape)    # (10000, 28, 28) 행무시
print(Y_test.shape)

from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import os
import tensorflow as tf

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255 # (60000, 28, 28, 1) convolution layer로 바꾸기위해 와꾸 맞춤
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255 # (10000, 28, 28, 1) convolution layer로 바꾸기위해 와꾸 맞춤

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
print(Y_train[0])
print(Y_train.shape)
print(Y_test.shape)

from sklearn.model_selection import train_test_split
X1_train, X2_train, Y1_train, Y2_train = train_test_split(X_train, Y_train, random_state=66, test_size=0.5)
X1_test, X2_test, Y1_test, Y2_test = train_test_split(X_test, Y_test, random_state=66, test_size=0.5)

print(X1_train.shape) # (30000, 28, 28, 1)
print(Y1_train.shape) # (30000, 10)
print(X1_test.shape) # (5000, 28, 28, 1)
print(Y1_test.shape) # (5000, 10)

# 컨블루션 신경망의 설정
input1 = Input(shape=(28,28,1))
layer1 = Conv2D(32, kernel_size=(3,3), activation='relu')(input1)
layer2 = Conv2D(64, (3,3), activation='relu')(layer1)
layer3 = MaxPooling2D(pool_size=2)(layer2)
layer4 = Dropout(0.25)(layer3)
layer5 = Flatten()(layer4)
layer6 = Dense(128, activation='relu')(layer5)
layer7 = Dropout(0.5)(layer6)
middle1 = Dense(1)(layer7)

input2 = Input(shape=(28,28,1))
layer1 = Conv2D(32, kernel_size=(3,3), activation='relu')(input2)
layer2 = Conv2D(64, (3,3), activation='relu')(layer1)
layer3 = MaxPooling2D(pool_size=2)(layer2)
layer4 = Dropout(0.25)(layer3)
layer5 = Flatten()(layer4)
layer6 = Dense(128, activation='relu')(layer5)
layer7 = Dropout(0.5)(layer6)
middle2 = Dense(1)(layer7)

# 모델 합치기 concatenate
from keras.layers.merge import concatenate
merge1 = concatenate([middle1, middle2]) 

output1 = Dense(30)(merge1)
output1 = Dense(13)(output1)
output1 = Dense(10, activation='softmax')(output1)

output2 = Dense(15)(merge1)
output2 = Dense(32)(output2)
output2 = Dense(10, activation='softmax')(output2)

model = Model(inputs=[input1, input2], outputs=[output1, output2])
# model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# 분류모델에선 무조건 categorical_crossentropy 사용

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

# 모델의 실행
history = model.fit([X1_train, X2_train], [Y1_train, Y2_train], validation_data=([X1_test, X2_test], [Y1_test, Y2_test]),
                    epochs=2, batch_size=200, verbose=1, # epochs=30
                    callbacks=[early_stopping_callback]) #,checkpointer])

# 테스트 정확도 출력
print('\n Test Accuracy: %.4f' % (model.evaluate([X1_test, X2_test], [Y1_test, Y2_test])[1]))
