#using tensorflow and gym(for reinforcement open api studying)

import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt

# env = gym.make('MountainCar-v0')
# print(env.observation_space)
# print(env.observation_space.low)
# print(env.observation_space.high)
# print(env._max_episode_steps)
# print(env.action_space)

# env.reset()
# env.render() 실행결과를 화면으로 출력
# step = 0
# score = 0

# step을 진행(random한 action을 수행 여기선 0,1,2중 하나)
# obs(환경이 바뀐 상태), reward(보상), done(에피소드 종료 여부), info(기타 정보)
# while True:
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     score += reward
#     step += 1
#     if done:
#         break
# print('Final Score: ', score)
# print('Step: ', step)

# env.close()

env = gym.make('MountainCar-v0')
env.reset()

scores = []
training_data = []
accepted_scores = []
required_score = -198
for i in range(20000):
    env.reset()
    score = 0
    game_memory = []
    previous_obs = []

    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if len(previous_obs) > 0:
            game_memory.append([previous_obs, action])
        previous_obs = obs
        if obs[0] > -0.2:
            reward = 1
        score += reward
        if done:
            break
    scores.append(score)
    if score > required_score:
        accepted_scores.append(score)
        for data in game_memory:
            training_data.append(data)

train_x = np.array([i[0] for i in training_data]).reshape(-1, 2)
train_y = np.array([i[1] for i in training_data]).reshape(-1, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(2, ), activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(train_x, train_y, epochs=30, callbacks=[callback], batch_size=16, validation_split=0.25)

plt.plot(history.history['accuracy'], 'b-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
plt.legend()
plt.show()

env.close()
env = gym.make('MountainCar-v0')
env.reset()

score = 0
step = 0
previous_obs = []
while True:
    if len(previous_obs) == 0:
        action = env.action_space.sample()
    else:
        logit = model.predict(np.expand_dims(previous_obs, axis=0))[0]
        action = np.argmax(logit)
    obs, reward, done, _ = env.step(action)
    previous_obs = obs
    score += reward
    step += 1
    if done:
        break
print('score : ', score)
print('step : ', step)
env.close()