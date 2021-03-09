import tensorflow as tf
import numpy as np
import math

print(tf.__version__)
rand = tf.random.uniform([1], 0, 1)
print(rand)

rand = tf.random.normal([1], 0, 1)
print(rand)

def sigmoid(x):
    return 1/(1+math.exp(-x))
def ReLU(x):
    return np.maximum(0, x)

# x = 1
# y = 0
# w = tf.random.normal([1], 0, 1)
# output = sigmoid(x*w)
# print(output)

# x = 1
# y = 0
# w = tf.random.normal([1], 0, 1)
# output = ReLU(x*w)
# print(output)

# x = 0
# y = 1
# w = tf.random.normal([1], 0, 1)
# a = 0.1
# for i in range(1000):
#     output = sigmoid(x*w)
#     err = y - output
#     w = w + x*a*err
#     if i % 100 == 99:
#         print(i, err, output)

# x = 0
# y = 1
# w = tf.random.normal([1], 0, 1)
# a = 0.1
# for i in range(1000):
#     output = ReLU(x*w)
#     err = y - output
#     w = w + x*a*err
#     if i % 100 == 99:
#         print(i, err[0], output[0])

# x = 0
# y = 1
# w = tf.random.normal([1], 0, 1)
# b = tf.random.normal([1], 0, 1)
# a = 0.1
# for i in range(1000):
#     output = sigmoid(x*w+b)
#     err = y - output
#     w = w + x*a*err
#     b = b + a*err
#     if i % 100 == 99:
#         print(i, err, output)

# x = 0
# y = 1
# w = tf.random.normal([1], 0, 1)
# b = tf.random.normal([1], 0, 1)
# a = 0.1
# for i in range(1000):
#     output = ReLU(x*w+b)
#     err = y - output
#     w = w + x*a*err
#     b = b + a*err
#     if i % 100 == 99:
#         print(i, err[0], output[0])

# x = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
# y = np.array([[1], [0], [0], [0]])
# w = tf.random.normal([2], 0, 1)
# b = tf.random.normal([1], 0, 1)
# a = 0.1

# for i in range(10000):
#     error_sum = 0
#     for j in range(4):
#         output = sigmoid(np.sum(x[j]*w)+b)
#         error = y[j][0] - output
#         w = w + x[j] * a * error
#         b = b + a * error
#         error_sum += error
#     if i % 1000 == 999:
#         print(i, error_sum)

# for i in range(4):
#     print('x:', x[i], 'Y:', y[i], 'Output', sigmoid(np.sum(x[i]*w)+b))

# x = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
# y = np.array([[1], [1], [1], [0]])
# w = tf.random.normal([2], 0, 1)
# b = tf.random.normal([1], 0, 1)
# a = 0.1

# for i in range(10000):
#     error_sum = 0
#     for j in range(4):
#         output = sigmoid(np.sum(x[j]*w)+b)
#         error = y[j][0] - output
#         w = w + x[j] * a * error
#         b = b + a * error
#         error_sum += error
#     if i % 1000 == 999:
#         print(i, error_sum)

# for i in range(4):
#     print('x:', x[i], 'Y:', y[i], 'Output', sigmoid(np.sum(x[i]*w)+b))

# x = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
# y = np.array([[0], [1], [1], [0]])
# w = tf.random.normal([2], 0, 1)
# b = tf.random.normal([1], 0, 1)
# a = 0.1

# for i in range(10000):
#     error_sum = 0
#     for j in range(4):
#         output = sigmoid(np.sum(x[j]*w)+b)
#         error = y[j][0] - output
#         w = w + x[j] * a * error
#         b = b + a * error
#         error_sum += error
#     if i % 1000 == 999:
#         print(i, error_sum)

# for i in range(4):
#     print('x:', x[i], 'Y:', y[i], 'Output', sigmoid(np.sum(x[i]*w)+b))

x = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
y = np.array([[0], [1], [1], [0]])
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=2, activation='sigmoid', input_shape=(2,)),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), loss='mse')
model.summary()
history = model.fit(x, y, epochs=2000, batch_size=1)
print(model.predict(x))

for weight in model.weights:
    print(weight)