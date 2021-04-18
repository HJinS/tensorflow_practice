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


class ResidualUnit(tf.keras.Model):
    def __init__(self, filter_in, filter_out):
        super(ResidualUnit, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filter_out, kernel_size=(1, 1), padding="same", activation="relu")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.av1 = tf.keras.layers.Activation(tf.nn.relu)

        self.conv2 = tf.keras.layers.Conv2D(filter_out, kernel_size=(3, 3), padding="same", activation="relu")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.av2 = tf.keras.layers.Activation(tf.nn.relu)
        
        self.conv3 = tf.keras.layers.Conv2D(filter_out, kernel_size=(1, 1), padding="same", activation="relu")
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.av3 = tf.keras.layers.Activation(tf.nn.relu)
        if filter_in == filter_out:
            self.identity = lambda x:x
        else:
            self.identity = tf.keras.layers.Conv2D(filter_out, (1,1), padding="same")
    
    def call(self, x, training=False, mask=None):
        h = self.conv1(x)
        h = self.bn1(h, training=training)
        h = self.av1(h)
        
        h = self.conv2(h)
        h = self.bn2(h, training=training)
        h = self.av2(h)

        h = self.conv3(h)
        h = self.bn3(h, training=training)
        h = self.identity(x) + h

        y = self.av3(h)
        return y

class ResnetLayer(tf.keras.Model):
    def __init__(self, filter_in, filters):
        super(ResnetLayer, self).__init__()
        self.sequnce = list()

        for f_in, f_out in zip([filter_in] + list(filters), filters):
            self.sequnce.append(ResidualUnit(f_in, f_out))
    
    def call(self, x, training=False, mask=None):
        for unit in self.sequnce:
            x = unit(x, training=training)
        return x

# ((input - kernel + 2 * padding) / stride) + 1 = conv output size
# (input / pooling) = pooling output size

class ResNet(tf.keras.Model):
    def __init__(self):
        super(ResNet, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(32, (7,7), padding="same", activation="relu", input_shape=(224, 224, 3), strides=2) #224x224x32
        self.pool1 = tf.keras.layers.MaxPool2D((2,2)) #112x112x32

        self.res1 = ResnetLayer(32, (64, 64)) #112x112x64
        self.res2 = ResnetLayer(64, (64, 64)) #112x112x64

        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(1,1), strides=(2,2)) #56x56x64

        self.res3 = ResnetLayer(64, (128, 128)) #56x56x128
        self.res4 = ResnetLayer(128, (128, 128)) #56x56x128

        self.conv3 = tf.keras.layers.Conv2D(128, kernel_size=(1,1), strides=(2,2)) #28x28x128


        self.res5 = ResnetLayer(128, (256, 256)) #28x28x256
        self.res6 = ResnetLayer(256, (256, 256)) #28x28x256
        self.res7 = ResnetLayer(256, (256, 256)) #28x28x256

        self.conv4 = tf.keras.layers.Conv2D(256, kernel_size=(1,1), strides=(2,2)) #14x14x256

        self.res8 = ResnetLayer(256, (512, 512)) #14x14x512
        self.res9 = ResnetLayer(512, (512, 512)) #14x14x512
        self.res10 = ResnetLayer(512, (512, 512)) #14x14x512

        self.conv5 = tf.keras.layers.Conv2D(512, kernel_size=(1,1), strides=(2,2)) #7x7x512

        self.res11 = ResnetLayer(512, (1024, 1024)) #7x7x1024
        self.res12 = ResnetLayer(1024, (1024, 1024)) #7x7x1024

        self.pool2 = tf.keras.layers.GlobalAveragePooling2D()

        self.dense1 = tf.keras.layers.Dense(1024, activation="relu")
        self.dense2 = tf.keras.layers.Dense(100, activation="softmax")

    def call(self, x, training=False, mask=None):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.res1(x, training=training)
        x = self.res2(x, training=training)

        
        x = self.conv2(x)

        x = self.res3(x, training=training)
        x = self.res4(x, training=training)


        x = self.conv3(x)

        x = self.res5(x, training=training)
        x = self.res6(x, training=training)
        x = self.res7(x, training=training)

        x = self.conv4(x)

        x = self.res8(x, training=training)
        x = self.res9(x, training=training)
        x = self.res10(x, training=training)

        x = self.conv5(x)

        x = self.res11(x, training=training)
        x = self.res12(x, training=training)

        x = self.pool2(x)
        x = self.dense1(x)
        return self.dense2(x)



class ResidualUnit(tf.keras.Model):
    def __init__(self, filter_in, filter_out):
        super(ResidualUnit, self).__init__()
        filter_in = filter_out // 4
        self.conv1 = tf.keras.layers.Conv2D(filter_in, kernel_size=(1, 1), padding="same", activation="relu")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.av1 = tf.keras.layers.Activation(tf.nn.relu)

        self.conv2 = tf.keras.layers.Conv2D(filter_in, kernel_size=(3, 3), padding="same", activation="relu")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.av2 = tf.keras.layers.Activation(tf.nn.relu)
        
        self.conv3 = tf.keras.layers.Conv2D(filter_out, kernel_size=(1, 1), padding="same", activation="relu")
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.shortcut = self._shortcut(filter_in, filter_out)
        self.add = tf.keras.layers.Add()
        self.av3 = tf.keras.layers.Activation(tf.nn.relu)
    
    def call(self, x, training=False, mask=None):
        h = self.conv1(x)
        h = self.bn1(h, training=training)
        h = self.av1(h)
        
        h = self.conv2(h)
        h = self.bn2(h, training=training)
        h = self.av2(h)

        h = self.conv3(h)
        h = self.bn3(h, training=training)
        shortcut = self.shortcut(x)
        h = self.add([h, shortcut])
        y = self.av3(h)

        return y
    def _shortcut(self, filter_in, filter_out):
        if filter_in == filter_out:
            return lambda x : x
        else:
            return self._projection(filter_out)

    def _projection(self, filter_out):
        return tf.keras.layers.Conv2D(filter_out, kernel_size=(1, 1), padding="same")

# ((input - kernel + 2 * padding) / stride) + 1 = conv output size
# (input / pooling) = pooling output size

class ResNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(7, 7), input_shape=(224, 224, 3), strides=2) #224x224x64
        self.bn = tf.keras.layers.BatchNormalization()


        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2) #112x112x64

        self.res1 = ResidualUnit(64, 256) #56x56x256
        self.res2 = ResidualUnit(256, 256) #56x56x256
        self.res3 = ResidualUnit(256, 256) #56x56x256

        self.conv2 = tf.keras.layers.Conv2D(512, kernel_size=(1,1), strides=2) #28x28x512

        self.res4 = ResidualUnit(512, 512) #28x28x512
        self.res5 = ResidualUnit(512, 512) #28x28x512
        self.res6 = ResidualUnit(512, 512) #28x28x512
        self.res7 = ResidualUnit(512, 512) #28x28x512

        self.conv3 = tf.keras.layers.Conv2D(1024, kernel_size=(1,1), strides=2) #14x14x1024

        self.res8 = ResidualUnit(1024, 1024) #14x14x1024
        self.res9 = ResidualUnit(1024, 1024) #14x14x1024
        self.res10 = ResidualUnit(1024, 1024) #14x14x1024
        self.res11 = ResidualUnit(1024, 1024) #14x14x1024
        self.res12 = ResidualUnit(1024, 1024) #14x14x1024
        self.res13 = ResidualUnit(1024, 1024) #14x14x1024

        self.conv4 = tf.keras.layers.Conv2D(2048, kernel_size=(1,1), strides=2) #7x7x2048

        self.res14 = ResidualUnit(2048, 2048) #7x7x2048
        self.res15 = ResidualUnit(2048, 2048) #7x7x2048
        self.res16 = ResidualUnit(2048, 2048) #7x7x2048

        self.pool2 = tf.keras.layers.GlobalAveragePooling2D()

        self.dense1 = tf.keras.layers.Dense(1000, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(100, activation=tf.nn.softmax)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)

        x = self.pool1(x)

        x = self.res1(x, training=training)
        x = self.res2(x, training=training)
        x = self.res3(x, training=training)
        
        x = self.conv2(x)

        x = self.res4(x, training=training)
        x = self.res5(x, training=training)
        x = self.res6(x, training=training)
        x = self.res7(x, training=training)

        x = self.conv3(x)

        x = self.res8(x, training=training)
        x = self.res9(x, training=training)
        x = self.res10(x, training=training)
        x = self.res11(x, training=training)
        x = self.res12(x, training=training)
        x = self.res13(x, training=training)

        x = self.conv4(x)

        x = self.res14(x, training=training)
        x = self.res15(x, training=training)
        x = self.res16(x, training=training)

        x = self.pool2(x)
        x = self.dense1(x)
        return self.dense2(x)


train_template = "train - Epoch {}, Loss: {}, Accuracy: {}"
print(train_template.format(epoch+1, train_loss.result(), train_accuracy.result() * 100))
test_template = "test - Loss: {}, Accuracy: {}"
print(test_template.format(test_loss.result(), test_accuracy.result() * 100))

    x_data = tfio.IODataset.from_hdf5("data_set1.hdf5", dataset="/x_data")
    y_data = tfio.IODataset.from_hdf5("data_set1.hdf5", dataset="/y_data")

    train = tf.data.Dataset.zip((x_data, y_data))
    train.shuffle(100009)
    test = train.take(20000)
    train.shuffle(100009)
    test.shuffle(20000)

    train = train.batch(BATCH_SIZE)
    test = test.batch(BATCH_SIZE)


def get_dataset(img_dir):
    dir_list = os.listdir(img_dir)
    file = h5py.File('data_set.hdf5', 'w')
    x_dataset = file.create_dataset("x_data", (100009, 224, 224, 3), dtype='float32', maxshape=(None, 224, 224, 3))
    y_dataset = file.create_dataset("y_data", (100009, 100), dtype='int', maxshape=(None, 100))
    idx = 0
    gif_cnt = 0

    img_generator = image.ImageDataGenerator(
            rotation_range = 20,
            zoom_range = 0.10,
            shear_range = 0.1,
            width_shift_range = 0.10,
            height_shift_range = 0.10,
            vertical_flip = True)
    
    categories = []
    for e in dir_list:
        ans = e
        img_loc = os.path.join(img_dir, e)
        img_list = os.listdir(img_loc)
        categories.append(ans)

        for file in img_list:
            fin_loc = os.path.join(img_loc, file)
            _, ext = os.path.splitext(fin_loc)
            if ext == '.gif' or ext == '.GIF':
                gif_cnt += 1
                continue
            img = image.load_img(fin_loc, target_size=(224, 224))
            img_tensor = image.img_to_array(img)
            img_tensor = img_tensor / 255.0
            img_tensor = img_generator.random_transform(img_tensor)
            label_classes = [0] * 100
            label_classes[len(categories)-1] = 1
            x_dataset[idx] = img_tensor
            y_dataset[idx] = label_classes
            idx += 1
            if idx % 1000 == 0:
                print("idx = ", idx)
            
    print("idx = ", idx)
    print("git_img = ", gif_cnt)

train_template = "train - Epoch {}, Loss: {}, Accuracy: {}"
print(train_template.format(epoch+1, train_loss.result(), train_accuracy.result() * 100), "step = ", step, "img_size = ", np.shape(images), "label_size = ", np.shape(labels))

test_template = "test - Loss: {}, Accuracy: {}"
print(test_template.format(test_loss.result(), test_accuracy.result() * 100), "step = ", step, "img_size = ", np.shape(test_images), "label_size = ", np.shape(test_labels))