from keras.models import Sequential

filter_size = 32
kernel_size = (3,3)

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
model = Sequential()
model.add(Conv2D(7, (2,2), padding='same',
                  input_shape=(28,28,1))) #7은 첫번째 레이어의 아웃풋값 / 픽셀, 가로, 세로 흑백(컬러) 10, 10, 1(3) / cnn의 특징은 특징(feature)을 잡는것
# input_shape=(None, 가로픽셀, 세로픽셀, feature) 1흑백(깊이1) 3컬러 None개 사진을 훈련
# model.add(Conv2D(16, (2,2)))
# model.add(MaxPooling2D(3,3))
# model.add(Conv2D(8, (2,2)))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(10))

model.summary()
'''
padding='same'
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 28, 28, 7)         35
_________________________________________________________________
flatten_1 (Flatten)          (None, 5488)              0
_________________________________________________________________
dense_1 (Dense)              (None, 10)                54890
_________________________________________________________________
dense_2 (Dense)              (None, 10)                110
=================================================================
Total params: 55,035
Trainable params: 55,035
Non-trainable params: 0
'''
'''
padding='same'안했을 때
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 27, 27, 7)         35
_________________________________________________________________
flatten_1 (Flatten)          (None, 5103)              0
_________________________________________________________________
dense_1 (Dense)              (None, 10)                51040
_________________________________________________________________
dense_2 (Dense)              (None, 10)                110
=================================================================
Total params: 51,185
Trainable params: 51,185
Non-trainable params: 0
_________________________________________________________________
'''
# keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), 
#     padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, 
#     use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', 
#     kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
#     kernel_constraint=None, bias_constraint=None)

# filters나가는 것,아웃풋/kernel_size 필터사이즈,자르는크기/strides(1,1)한칸씩 이동
# ex 6*6사이즈를 kernel_size(2,2) stride(2,2) 면 -> 2*2가 나옴
# 5*5그림에서 kernel_size(1,1) -> 5*5 // (2,2) -> 4*4
# padding의default valid// padding='same' 원래있던 이미지에 가로세로 한줄을 씌운다(0or1). 원래있던shape과 동일하게

# 자르는 과정 이해
# model.add(Conv2D(7, (3,3), input_shape=(5,5,1)))
# model.add(Flatten())
# Conv2D에서 5*5 그림을 3*3으로 자르면 다음 레이어(Flatten)으로 3*3짜리 7장이 나감 -> (3,3,7) 가로,세로,필터
# padding='same' -> 위,아래,가로,세로로 한줄씌워서 다음레이어(Flatten)으로 5*5내보냄 -> (5,5,7)

# Conv2D는 나가는 값이 같다. return_sequence필요없음 그대로 다음 레이어에 Conv2D적용 가능
# model.add(Conv2D(3, (2,2), input_shape=(5,5,1))) -> (4,4,3)
# model.add(Conv2D(4, (2,2)) 

# padding하는 이유 가장자리 픽셀데이터의 손실을 줄이기 위함

# Shape맞추기 -> Dimension맞추기
# model.add(Dense(10, input_shape=(5,)))
# ->  (5,10,1)로 reshape필요
# model.add(LSTM(10, input_shape=(5,1)))
# -> (5,10,1,1)로 reshape필요
# model.add(Conv2D(10,(2,2), input_shape=(28,28,1)))

# 레이어 모델끼리 서로 shape을 바꿔보며 상호호환해야한다.
# Conv2D -> DNN 예측값은 잘 맞을 수도 있지만 DNN -> Conv2D는 잘 안될수도