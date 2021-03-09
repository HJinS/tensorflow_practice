import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

#이미지 전처리 뒤집기(horizontal flip), 회전(rotate), 기울임(shear), 확대(zoom), 평행이동(shift)
img_generator = ImageDataGenerator(
    rotation_range = 10,
    room_range = 0.10,
    shear_range = 0.5,
    width_shift_range = 0.10,
    height_shift_range = 0.10,
    horizontal_flip = True,
    vertical_flip = False
)
augment_size = 100
x_augmented = img_generator.flow(
    np.tile(train_X[0].reshape(28*28), 100).reshape(-1, 28, 28, 1),
    np.zeros(augment_size)
    batch_size = augment_size,
    shuffle = False).next()[0]

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(input_shape=(28, 28, 1), kernel_size=(3,3), filters=16),
    tf.keras.layers.MaxPool2D(strides=(2,2)),
    tf.keras.layers.Conv2D(kernel_size=(3, 3), filter=32),
    tf.keras.layers.MaxPool2D(strides=(2,2)),
    tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dropout(rate=0.3),
    tf.keras.layers.Dense(units=10, activation='softmax')
])