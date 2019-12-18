from keras.models import Sequential

filter_size = 32
kernel_size = (3,3)

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
model = Sequential()
model.add(Conv2D(3, (3,3), padding='same',
                  input_shape=(28,28,1))) #7은 첫번째 레이어의 아웃풋값 /None, 픽셀, 가로, 세로 흑백(컬러) 10, 10, 1(3) / cnn의 특징은 특징(feature)을 잡는것 lstm보다는 느림

model.add(Conv2D(4, (2,2)))
model.add(Conv2D(16, (2,2)))
model.add(Conv2D(8, (2,2)))
# model.add(MaxPooling2D(3,3))
model.add(Flatten()) # cnn을 dense모델로 전환하기위해선 flatten해야함
model.add(Dense(10))
model.add(Dense(1))

model.summary()
'''
padding='same'
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 28, 28, 3)         30
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 27, 27, 4)         52
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 26, 26, 16)        272
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 25, 25, 8)         520
_________________________________________________________________
flatten_1 (Flatten)          (None, 5000)              0
_________________________________________________________________
dense_1 (Dense)              (None, 10)                50010
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 11
=================================================================
Total params: 50,895
Trainable params: 50,895
Non-trainable params: 0
'''
'''
padding='same'안했을 때
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 26, 26, 3)         30
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 25, 25, 4)         52
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 24, 24, 16)        272
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 23, 23, 8)         520
_________________________________________________________________
flatten_1 (Flatten)          (None, 4232)              0
_________________________________________________________________
dense_1 (Dense)              (None, 10)                42330
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 11
=================================================================
Total params: 43,215
Trainable params: 43,215
Non-trainable params: 0
'''
