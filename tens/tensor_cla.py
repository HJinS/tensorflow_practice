import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

mushrooms = pd.read_csv('mushrooms.csv')
mushrooms.head()
labelencoder = LabelEncoder()
for col in mushrooms.columns:
    mushrooms[col] = labelencoder.fit_transform(mushrooms[col])
mushrooms.head()

y = mushrooms['class'].values
x = mushrooms.drop(['class'], axis=1)
x = x.values
x = (x - x.min()) / (x.max() - x.min())
print(np.shape(x))
#train_test_split 테스트 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#to_categorical one-hot encoding형태로 나오도록
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=48, activation='relu', input_shape=(22,)),
    tf.keras.layers.Dense(units=24, activation='relu'),
    tf.keras.layers.Dense(units=12, activation='relu'),
    tf.keras.layers.Dense(units=2, activation='sigmoid')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.07),
                loss='binary_crossentropy', metrics=['accuracy'])

#다항분류시 categorical_crossentropy를 사용
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=48, activation='relu', input_shape=x_train[0].shape),
#     tf.keras.layers.Dense(units=24, activation='relu'),
#     tf.keras.layers.Dense(units=12, activation='relu'),
#     tf.keras.layers.Dense(units=3, activation='softmax')
# ])

# model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.07),
#                 loss='categorical_crossentropy', metrics=['accuracy'])


#callback.EarlyStopping를 사용해서 val_loss가 3번 이상 연속으로 증가한다면 멈추고 최저의 loss를 사용
history = model.fit(x_train, y_train, epochs=25, batch_size=32, validation_split=0.25, callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')])

print(model.predict(x_test[0:1]))