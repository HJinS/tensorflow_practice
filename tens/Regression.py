import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Linear Regression(선형 회귀)
# numpy
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

# plt.plot(x, y, 'bo')
# plt.show()

x_mean = sum(x) / len(x)
y_mean = sum(y) / len(y)

a = sum([(x_ - x_mean) * (y_ - y_mean) for x_, y_ in list(zip(x, y))])
a /= sum([(x_ - x_mean) ** 2 for x_ in x])
b = y_mean - a * x_mean
print('a:', a, 'b:', b)

x_pred = np.arange(min(x), max(x), 0.01)
y_pred = a * x_pred + b

# plt.plot(x_pred, y_pred, 'r-')
# plt.plot(x, y, 'bo')
# plt.show()

# tensorflow
w = tf.Variable(random.random())
b = tf.Variable(random.random())

def residue():
    y_pred = w * x + b
    re = tf.reduce_mean((y - y_pred) ** 2)
    return re

optimizer = tf.optimizers.Adam(lr=0.07)
for i in range(1000):
    optimizer.minimize(residue, var_list=[w, b])

    if i % 100 == 99:
        print(i, 'w:', w.numpy(), 'b:', b.numpy(), 'loss:', residue().numpy())

x_pred = np.arange(min(x), max(x), 0.01)
y_pred = w * x_pred + b

# plt.plot(x_pred, y_pred, 'r-')
# plt.plot(x, y, 'bo')
# plt.show()

# sklearn

lin_reg = LinearRegression()
lin_reg.fit(x, y)
print(lin_reg.intercept_, lin_reg.coef_)

x_pred = np.array([[0], [2]])
y_pred = lin_reg.predict(x_pred)

# plt.plot(x, y, 'bo')
# plt.plot(x_pred, y_pred, 'r-')
# plt.show()

# Polynomial Regression(다항 회귀)

x = 2 * np.random.rand(100, 1)
y = 6 * (x ** 2) + 3 * x + 4 + np.random.randn(100, 1)
plt.plot(x, y, 'bo')
plt.show()

# tensorflow

w = tf.Variable(random.random())
b = tf.Variable(random.random())
c = tf.Variable(random.random())

def residue():
    y_pred = w * x ** 2 + b * x + c
    loss = tf.reduce_mean((y - y_pred) ** 2)
    return loss

optimizer = tf.optimizers.Adam(lr=0.07)
for i in range(1000):
    optimizer.minimize(residue, var_list=[w,b,c])
    if i % 100 == 99:
        print(i, 'w:', w.numpy(), 'b:', b.numpy(), 'c:', c.numpy(), 'loss:', residue().numpy())

x_pred = np.arange(min(x), max(x), 0.01)
y_pred = w * x_pred * x_pred + b * x_pred + c

plt.plot(x_pred, y_pred, 'r-')
plt.plot(x, y, 'bo')
plt.show()

# sklearn

poly_features = PolynomialFeatures(degree=2, include_bias=True)
x_poly = poly_features.fit_transform(x)

lin_reg = LinearRegression()
lin_reg.fit(x_poly, y)

x_new = np.linspace(0, 2, 100).reshape(100, 1)
x_new_poly = poly_features.transform(x_new)
y_predict = lin_reg.predict(x_new_poly)

plt.plot(x_new, y_predict, 'r-')
plt.plot(x, y, 'bo')
plt.show()