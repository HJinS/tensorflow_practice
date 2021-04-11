import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import shutil, os, h5py, datetime
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import normalize
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.abspath('..')
img_dir = os.path.join(BASE_DIR,"Food_data\dataset")
print()
print(img_dir)

def get_dataset(img_dir):
    dir_list = os.listdir(img_dir)
    file = h5py.File('data_set.hdf5', 'w')
    x_dataset = file.create_dataset("x_data", (100009, 224, 224, 3), dtype='float32', maxshape=(None, 224, 224, 3))
    y_dataset = file.create_dataset("y_data", (100009, 100), dtype='int', maxshape=(None, 100))
    idx = 0
    gif_cnt = 0

    img_generator = image.ImageDataGenerator(
            rotation_range = 20,
            zoom_range = 0.20,
            shear_range = 0.3,
            width_shift_range = 0.30,
            height_shift_range = 0.30,
            horizontal_flip = True,
            vertical_flip = True)

    for e in dir_list:
        ans = e
        categories = []
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

def fit_and_save():
    global BASE_DIR
    model = ResNet()

    EPOCHS = 10
    BATCH_SIZE = 100
    
    save_dir = "Food_CNN\Training1\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_save_path = os.path.join(BASE_DIR, save_dir)

    train_log_dir = "logs/train" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    test_log_dir = "logs/test" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    x_data = tfio.IODataset.from_hdf5("data_set.hdf5", dataset="/x_data")
    y_data = tfio.IODataset.from_hdf5("data_set.hdf5", dataset="/y_data")

    train = tf.data.Dataset.zip((x_data, y_data))
    train.shuffle(100009)
    test = train.take(20000)
    train.experimental.shuffle_and_repeat(100009, 4)
    test.experimental.shuffle_and_repeat(100009, 4)

    train = train.batch(BATCH_SIZE, drop_remainder=True)
    test = test.batch(BATCH_SIZE, drop_remainder=True)
    
    for epoch in range(EPOCHS):
        for images, labels in train:
            train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy)
            train_template = "train - Epoch {}, Loss: {}, Accuracy: {}"
            print(train_template.format(epoch+1, train_loss.result(), train_accuracy.result() * 100))
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result() * 100, step=epoch)
            

        for test_images, test_labels in test:
            test_step(model, test_images, test_labels, loss_object, test_loss, test_accuracy)
            test_template = "test - Loss: {}, Accuracy: {}"
            print(test_template.format(test_loss.result(), test_accuracy.result() * 100))
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result() * 100, step=epoch)
    
        template = "Epoch {}, Loss: {}, Accuracy: {}, TestLoss: {}, Test Accuracy: {}"
        print(template.format(epoch+1, train_loss.result(), train_accuracy.result() * 100, test_loss.result(), test_accuracy.result() * 100))

    model.save_weights(model_save_path)



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

@tf.function
def train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
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
    except RuntimeError as e:
        print("---------------ERROR-----------------")
        print(e)
        print("---------------ERROR-----------------")

with tf.device('/gpu:0'):
    fit_and_save()