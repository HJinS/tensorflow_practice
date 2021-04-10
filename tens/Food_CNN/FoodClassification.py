import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import shutil, os, random, h5py, datetime
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import normalize
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.abspath('..')
img_dir = os.path.join(BASE_DIR,"Food_data\dataset")

def get_dataset(img_dir):
    dir_list = os.listdir(img_dir)
    file = h5py.File('data_set.hdf5', 'w')
    x_dataset = file.create_dataset("x_data", (100009, 256, 256, 3), dtype='float32', maxshape=(None, 256, 256, 3))
    y_dataset = file.create_dataset("y_data", (100009, 100), dtype='int', maxshape=(None, 100))
    idx = 0
    gif_cnt = 0

    img_generator = image.ImageDataGenerator(
            rotation_range = 10,
            zoom_range = 0.10,
            shear_range = 0.2,
            width_shift_range = 0.10,
            height_shift_range = 0.10,
            horizontal_flip = True,
            vertical_flip = True)

    for e in dir_list:
        ans = e
        categories = []
        img_loc = os.path.join(img_dir, e)
        img_list = os.listdir(img_loc)
        categories.append(ans)

        for file in img_list:
            rand_num = random.randrange(1,6)
            fin_loc = os.path.join(img_loc, file)
            _, ext = os.path.splitext(fin_loc)
            if ext == '.gif' or ext == '.GIF':
                gif_cnt += 1
                continue
            img = image.load_img(fin_loc, target_size=(256, 256))
            img_tensor = image.img_to_array(img)
            img_tensor = img_tensor / 255.0
            if rand_num % 2 == 0:
                img_tensor = img_generator.random_transform(img_tensor, seed=2021)
            label_classes = [0] * 100
            label_classes[len(categories)-1] = 1
            x_dataset[idx] = img_tensor
            y_dataset[idx] = label_classes
            idx += 1
            if idx % 1000 == 0:
                print("idx = ", idx)
            
    print("idx = ", idx)
    print("git_img = ", gif_cnt)

def fit_and_save():

    model = ResNet()

    EPOCHS = 25
    BATCH_SIZE = 1000
    
    model_save_path = os.path.join(BASE_DIR,"training1" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.07)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalCrossentropy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalCrossentropy(name='test_accuracy')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    x_data = tfio.IODataset.from_hdf5("data_set.hdf5", dataset="/x_data")
    y_data = tfio.IODataset.from_hdf5("data_set.hdf5", dataset="/y_data")

    train = tf.data.Dataset.zip((x_data, y_data))
    train.shuffle(buffer_size = 2048)
    test = train.take(20000)

    train = train.batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    test = test.batch(BATCH_SIZE//10, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    print(train)
    model.compile(optimizer=optimizer, loss=loss_object, metrics=['accuracy'])
    train_history = model.fit(train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'), tensorboard_callback])

    test_history = model.predict(test, epochs=EPOCHS, batch_size=BATCH_SIZE, use_multiprocessing=True, steps=100, callbacks=[tensorboard_callback])

    model.save_weights(model_save_path)



class ResidualUnit(tf.keras.Model):
    def __init__(self, filter_in, filter_out, kernel_size):
        super(ResidualUnit, self).__init__()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(filter_out, kernel_size, padding="same")

        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filter_out, kernel_size, padding="same")

        if filter_in == filter_out:
            self.identity = lambda x:x
        else:
            self.identity = tf.keras.layers.Conv2D(filter_out, (1,1), padding="same")
    
    def call(self, x, training=False, mask=None):
        h = self.bn1(x, training=training)
        h = tf.nn.relu(h)
        h = self.conv1(h)

        h = self.bn2(h, training=training)
        h = tf.nn.relu(h)
        h = self.conv2(h)
        return self.identity(x) + h

class ResnetLayer(tf.keras.Model):
    def __init__(self, filter_in, filters, kernel_size):
        super(ResnetLayer, self).__init__()
        self.sequnce = list()

        for f_in, f_out in zip([filter_in] + list(filters), filters):
            self.sequnce.append(ResidualUnit(f_in, f_out, kernel_size))
    
    def call(self, x, training=False, mask=None):
        for unit in self.sequnce:
            x = unit(x, training=training)
        return x

# ((input - kernel + 2 * padding) / stride) + 1 = conv output size
# (input / pooling) = pooling output size

class ResNet(tf.keras.Model):
    def __init__(self):
        super(ResNet, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(32, (7,7), padding="same", activation="relu", input_shape=(256, 256, 3), strides=2) #256x256x32
        self.pool1 = tf.keras.layers.MaxPool2D((2,2)) #128x128x32

        self.res1 = ResnetLayer(32, (64, 64), (3, 3)) #128x128x64
        self.res2 = ResnetLayer(64, (64, 64), (3, 3)) #128x128x64
        self.res3 = ResnetLayer(64, (64, 64), (3, 3)) #128x128x64

        self.pool2 = tf.keras.layers.MaxPool2D((2,2)) #64x64x64

        self.res4 = ResnetLayer(64, (128, 128), (3, 3)) #64x64x128
        self.res5 = ResnetLayer(128, (128, 128), (3, 3)) #64x64x128
        self.res6 = ResnetLayer(128, (128, 128), (3, 3)) #64x64x128

        self.pool3 = tf.keras.layers.MaxPool2D((2,2)) #32x32x128

        self.res7 = ResnetLayer(128, (256, 256), (3, 3)) #32x32x256
        self.res8 = ResnetLayer(128, (256, 256), (3, 3)) #32x32x256
        self.res9 = ResnetLayer(128, (256, 256), (3, 3)) #32x32x256
        self.res10 = ResnetLayer(128, (256, 256), (3, 3)) #32x32x256

        self.pool4 = tf.keras.layers.MaxPool2D((2,2)) #16x16x256

        self.res11 = ResnetLayer(256, (512, 512), (3, 3)) #16x16x512
        self.res12 = ResnetLayer(256, (512, 512), (3, 3)) #16x16x512
        self.res13 = ResnetLayer(256, (512, 512), (3, 3)) #16x16x512

        self.pool5 = tf.keras.layers.MaxPool2D((2,2)) #8x8x512

        self.res14 = ResnetLayer(512, (1024, 512), (3, 3)) #8x8x1024
        self.res15 = ResnetLayer(512, (1024, 512), (3, 3)) #8x8x1024

        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation="relu")
        self.dense2 = tf.keras.layers.Dense(512, activation="relu")
        self.dense3 = tf.keras.layers.Dense(100, activation="softmax")

    def call(self, x, training=False, mask=None):
        x = self.conv1(x)

        x = self.pool1(x)
        x = self.res1(x, training=training)
        x = self.res2(x, training=training)
        x = self.res3(x, training=training)
        
        x = self.pool2(x)

        x = self.res4(x, training=training)
        x = self.res5(x, training=training)
        x = self.res6(x, training=training)

        x = self.pool3(x)

        x = self.res7(x, training=training)
        x = self.res8(x, training=training)
        x = self.res9(x, training=training)
        x = self.res10(x, training=training)

        x = self.pool4(x)

        x = self.res11(x, training=training)
        x = self.res12(x, training=training)
        x = self.res13(x, training=training)

        x = self.pool5(x)

        x = self.res14(x, training=training)
        x = self.res15(x, training=training)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

@tf.function
def train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zop(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
    predictions = model(images, training=False)

    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

# get_dataset(img_dir)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs, ", len(logical_gpus), "Logical GPUs")
        with tf.device('/gpu:0'):
            fit_and_save()
    except RuntimeError as e:
        print(e)

