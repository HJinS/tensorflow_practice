import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=6, activation='tanh', input_shape=(1,)),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), loss='mse')
model.summary()

history = model.fit(x, y, epochs=100)
x_pred = np.arange(min(x), max(x), 0.01)
y_pred = model.predict(x_pred)

plt.plot(x, y, 'bo')
plt.plot(x_pred, y_pred, 'r-')
plt.show()
