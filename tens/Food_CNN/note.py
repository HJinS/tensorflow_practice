# for e in dir_list:
#     ans = e
#     img_loc = os.path.join(img_dir, e)
#     img_list = os.listdir(img_loc)
#     categories.append(ans)

#     for file in img_list:
#         fin_loc = os.path.join(img_loc, file)
#         _, ext = os.path.splitext(fin_loc)
#         if ext == '.gif' or ext == '.GIF':
#             gif_cnt += 1
#             continue
#         img = image.load_img(fin_loc, target_size=(250, 250))
#         img_tensor = image.img_to_array(img)
#         img_tensor = img_tensor / 255.0
#         label_classes = [0] * 100
#         label_classes[len(categories)-1] = 1
#         x_dataset[idx] = img_tensor
#         y_dataset[idx] = label_classes
#         idx += 1

    
# x_data = tfio.IOTensor.from_hdf5("data_set.hdf5", dataset="data_group/x_data")
# y_data = tfio.IOTensor.from_hdf5("data_set.hdf5", dataset="data_group/y_data")

# x_data = HDF5Matrix(file_path, 'x_data')
# y_data = HDF5Matrix(file_path, 'y_data')
# x_train = HDF5Matrix(file_path, 'data_group/x_data', end=split_pos)
# x_test = HDF5Matrix(file_path, 'data_group/x_data', start=split_pos)
# y_train = HDF5Matrix(file_path, 'data_group/y_data', end=split_pos)
# y_test = HDF5Matrix(file_path, 'data_group/y_data', start=split_pos)


# print(np.shape(x_data))
# print(np.shape(x_train))
# print(np.shape(x_test))
# print(np.shape(y_train))
# print(np.shape(y_test))

def build_model(self):
        self.model = tf.keras.Sequential([
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
    def compile_and_fit(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.model.fit(train, epochs=5, validation_data(test), callbacks=[tensorboard_callback])

for epoch in range(EPOCHS):
  for images, labels in train_ds:
    train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy)

  for test_images, test_labels in test_ds:
    test_step(model, test_images, test_labels, loss_object, test_loss, test_accuracy)

  template = "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}"
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result() * 100,
                        test_loss.result(),
                        test_accuracy.result() * 100))

model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])
    train_history = model.fit(train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, monitor='loss'), tensorboard_callback_training])

    test_history = model.predict(test, batch_size=BATCH_SIZE, use_multiprocessing=True, steps=100, callbacks=[tensorboard_callback_test])