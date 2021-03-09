import tensorflow as tf
import numpy as np
# x = []
# y = []
# for i in range(6):
#     lst = list(range(i, i+4))
#     x.append(list(map(lambda c: [c/10], lst)))
#     y.append((i+4)/10)
# x = np.array(x)
# y = np.array(y)

# model = tf.keras.Sequential([
#     tf.keras.layers.SimpleRNN(units=10, return_sequences=False, input_shape=[4,1]),
#     tf.keras.layers.Dense(1)
# ])
# model.compile(optimizer='adam', loss='mse')
# history = model.fit(x, y, epochs=100, verbose=0)

# LSTM
# model = tf.keras.Sequential([
#     tf.keras.layers.LSTM(units=30, return_sequences=True, input_shape=[100, 2]),
#     tf.keras.layers.LSTM(units=30),
#     tf.keras.layers.Dense(1)
# ])
# model.compile(optimizer='adam', loss='mse')
# history = model.fit(x, y, epochs=100, verbose=0)

# GRU
# model = tf.keras.Sequential([
#     tf.keras.layers.GRU(units=30, return_squences=True, input_shape=[100, 2]),
#     tf.keras.layers.GRU(units=30),
#     tf.keras.layers.Dense(1)
# ])
# model.compile(optimizer='adam', loss='mse')
# history = model.fit(x, y, epochs=100, verbose=0)

# 자연어 전처리
# 자연어를 의미있는 숫자 벡터로
t = tf.keras.preprocessing.text.Tokenizer()
fit_text = "I love kanon so much"
t.fit_on_texts([fit_text])

test_text = "I like kanon so much"
squences = t.texts_to_sequences([test_text])

print("squences : ", squences)
print("word_index : ", t.word_index)

# 모델을 학습시키기 위해서는 길이가 일정해야 하는 경우가 많다 이때 padding을 넣어 길이를 유지
# padding = 'pre'인 경우는 앞에 0을 넣고, padding = 'post'인 경우는 0을 뒤에 넣는다
tf.keras.preprocessing.sequence.pad_sequence([[1, 2, 3], [3, 4, 5, 6], [7, 8]], maxlen=3, padding='pre')

# word embedding 자연어 처리에서 자연어를 수치화된 정보로 바꿈
# tf.keras.Embedding클래스를 사용하여 선언
# input_dim = 2000(단어의 크기 2000개의 단어 종류가 있다)
# output_dim = 128(임베딩의 출력 차원 주로 256, 512, 1024등)
# input_length = x.shape[1](일정한 입력 데이터의 크기)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(2000, 128, input_length=x.shape[1]),
    tf.keras.layers.SpatialDropout1D(0.4),
    tf.keras.layers.LSTM(196, dropout=0.2, recurrent_dropout=0.2),
    tf.keras.layers.Dense(2, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# print(model.predict(x))
# x_test = np.array([[[0.8], [0.9], [1.0], [1.1]]])
# print()
# print(model.predict(x_test))