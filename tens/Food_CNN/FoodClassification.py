import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import shutil, os, h5py, datetime
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import normalize
from sklearn.model_selection import train_test_split
import pickle


BASE_DIR = os.path.abspath('..')
img_dir = os.path.join(BASE_DIR,"Food_data\dataset")
print()
print(img_dir)

def get_dataset(train_dir, test_dir):
    train_list = os.listdir(train_dir)
    test_list = os.listdir(test_dir)
    train = h5py.File('train_set.hdf5', 'w')
    test =  h5py.File('test_set.hdf5', 'w')
    x_train = train.create_dataset("x_data", (80060, 224, 224, 3), dtype='float32', maxshape=(None, 224, 224, 3))
    y_train = train.create_dataset("y_data", (80060, 100), dtype='float32', maxshape=(None, 100))
    x_test = test.create_dataset("x_data", (19955, 224, 224, 3), dtype='float32', maxshape=(None, 224, 224, 3))
    y_test = test.create_dataset("y_data", (19955, 100), dtype='float32', maxshape=(None, 100))
    idx = 0

    img_generator = image.ImageDataGenerator(
            rotation_range = 5,
            zoom_range = 0.01,
            shear_range = 0.01,
            width_shift_range = 0.01,
            height_shift_range = 0.01)
    
    categories = []
    for e in train_list:
        ans = e
        img_loc = os.path.join(train_dir, e)
        img_list = os.listdir(img_loc)
        categories.append(ans)

        for file in img_list:
            fin_loc = os.path.join(img_loc, file)
            img = image.load_img(fin_loc, target_size=(224, 224))
            img_tensor = image.img_to_array(img)
            img_tensor = img_tensor / 255.
            img_tensor = img_generator.random_transform(img_tensor)
            label_classes = [0.0] * 100
            label_classes[len(categories)-1] = 1.0
            x_train[idx] = img_tensor
            y_train[idx] = label_classes
            idx += 1
            if idx % 1000 == 0:
                print("idx = ", idx)
    print("train over")      
    print("idx = ", idx)

    categories = []
    idx = 0
    for e in test_list:
        ans = e
        img_loc = os.path.join(test_dir, e)
        img_list = os.listdir(img_loc)
        categories.append(ans)

        for file in img_list:
            fin_loc = os.path.join(img_loc, file)
            img = image.load_img(fin_loc, target_size=(224, 224))
            img_tensor = image.img_to_array(img)
            img_tensor = img_tensor / 255.
            label_classes = [0.0] * 100
            label_classes[len(categories)-1] = 1.0
            x_test[idx] = img_tensor
            y_test[idx] = label_classes
            idx += 1
            if idx % 1000 == 0:
                print("idx = ", idx)
    print("test over")
    print("idx = ", idx)

def fit_and_save():
    global BASE_DIR
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.000004)

    model = ResNet((224, 224, 3), 100)
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()

    EPOCHS = 100
    BATCH_SIZE = 64
    
    save_dir = "Food_CNN\Training1\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_save_path = os.path.join(BASE_DIR, save_dir)
   
    summary_writer = tf.summary.create_file_writer(log_dir)
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    x_train = tfio.IODataset.from_hdf5("train_set.hdf5", dataset="/x_data")
    y_train = tfio.IODataset.from_hdf5("train_set.hdf5", dataset="/y_data")
    x_test = tfio.IODataset.from_hdf5("test_set.hdf5", dataset="/x_data")
    y_test = tfio.IODataset.from_hdf5("test_set.hdf5", dataset="/y_data")

    train = tf.data.Dataset.zip((x_train, y_train)).shuffle(80060).batch(BATCH_SIZE)
    test = tf.data.Dataset.zip((x_test, y_test)).shuffle(19955).batch(BATCH_SIZE)

    checkpoint_dir = "Checkpoint"
    checkpoint_path = os.path.join(BASE_DIR, "Food_CNN/" + checkpoint_dir)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer,net=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    for epoch in range(EPOCHS):
        for step, (images, labels) in enumerate(train):
            train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy)
            
        for step, (test_images, test_labels) in enumerate(test):
            test_step(model, test_images, test_labels, loss_object, test_loss, test_accuracy)
        
        with summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
            tf.summary.scalar('train_accuracy', train_accuracy.result() * 100, step=epoch)
            tf.summary.scalar('test_loss', test_loss.result(), step=epoch)
            tf.summary.scalar('test_accuracy', test_accuracy.result() * 100, step=epoch)
    
        template = "Epoch {}, Loss: {}, Accuracy: {}, TestLoss: {}, Test Accuracy: {}"
        print(template.format(epoch+1, train_loss.result(), train_accuracy.result() * 100, test_loss.result(), test_accuracy.result() * 100))
        ckpt.step.assign_add(1)
        if int(ckpt.step) % 10 == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        train_loss.reset_states()
        test_loss.reset_states()

    model.save(model_save_path)

def keep_training():

    EPOCHS = 100
    BATCH_SIZE = 64

    model = ResNet((224, 224, 3), 100)
    model.build(input_shape=(None, 224, 224, 3))
    checkpoint_dir = "Checkpoint"
    checkpoint_path = os.path.join(BASE_DIR, "Food_CNN/" + checkpoint_dir)
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.000004)
    save_dir = "Food_CNN\Training1\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_save_path = os.path.join(BASE_DIR, save_dir)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    ckpt.restore(manager.latest_checkpoint)

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    x_train = tfio.IODataset.from_hdf5("train_set.hdf5", dataset="/x_data")
    y_train = tfio.IODataset.from_hdf5("train_set.hdf5", dataset="/y_data")
    x_test = tfio.IODataset.from_hdf5("test_set.hdf5", dataset="/x_data")
    y_test = tfio.IODataset.from_hdf5("test_set.hdf5", dataset="/y_data")

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    summary_writer = tf.summary.create_file_writer(log_dir)

    train = tf.data.Dataset.zip((x_train, y_train)).shuffle(80060).batch(BATCH_SIZE)
    test = tf.data.Dataset.zip((x_test, y_test)).shuffle(19955).batch(BATCH_SIZE)

    last_step = int(ckpt.step)
    for epoch in range(last_step, EPOCHS):
        for step, (images, labels) in enumerate(train):
            train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy)
            
        for step, (test_images, test_labels) in enumerate(test):
            test_step(model, test_images, test_labels, loss_object, test_loss, test_accuracy)
        
        with summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss.result(), step=epoch)
            tf.summary.scalar('train_accuracy', train_accuracy.result() * 100, step=epoch)
            tf.summary.scalar('test_loss', test_loss.result(), step=epoch)
            tf.summary.scalar('test_accuracy', test_accuracy.result() * 100, step=epoch)
    
        template = "Epoch {}, Loss: {}, Accuracy: {}, TestLoss: {}, Test Accuracy: {}"
        print(template.format(epoch+1, train_loss.result(), train_accuracy.result() * 100, test_loss.result(), test_accuracy.result() * 100))
        ckpt.step.assign_add(1)
        if int(ckpt.step) % 10 == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        train_loss.reset_states()
        test_loss.reset_states()
    model.save(model_save_path)

class ResidualUnit(tf.keras.Model):
    def __init__(self, filter_in, filter_out):
        super().__init__()
        filter_n = filter_out // 4

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(filter_n, kernel_size=(1, 1), strides=1, padding="valid")
        self.dropout1 = tf.keras.layers.Dropout(0.5)

        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filter_n, kernel_size=(3, 3), strides=1, padding="same")
        self.dropout2 = tf.keras.layers.Dropout(0.5)

        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filter_out, kernel_size=(1, 1), strides=1, padding="valid")
        
        self.shortcut = self._shortcut(filter_in, filter_out)
    
    def call(self, x):
        
        h = self.bn1(x)
        h = tf.nn.relu(h)
        h = self.conv1(h)
        h = self.dropout1(h)

        h = self.bn2(h)
        h = tf.nn.relu(h)
        h = self.conv2(h)
        h = self.dropout2(h)

        h = self.bn3(h)
        shortcut = self.shortcut(x)
        h = tf.nn.relu(h)
        h = self.conv3(h)
        y = h + shortcut
        
        return y

    def _shortcut(self, filter_in, filter_out):
        if filter_in == filter_out:
            return lambda x : x
        else:
            return self._projection(filter_out)

    def _projection(self, filter_out):
        return tf.keras.layers.Conv2D(filter_out, kernel_size=(1, 1), padding="valid")

# ((input - kernel + 2 * padding) / stride) + 1 = conv output size
# (input / pooling) = pooling output size

class ResNet(tf.keras.Model):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(64, input_shape=input_shape, kernel_size=(7, 7), strides=2) #224x224x64
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

        self.dense1 = tf.keras.layers.Dense(700, activation=tf.nn.relu)
        self.dropout1 = tf.keras.layers.Dropout(0.7)
        self.dense2 = tf.keras.layers.Dense(400, activation=tf.nn.relu)
        self.dropout2 = tf.keras.layers.Dropout(0.7)
        self.dense3 = tf.keras.layers.Dense(output_dim, activation=tf.nn.softmax)

    def call(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = tf.nn.relu(x)

        x = self.pool1(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        x = self.conv2(x)

        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)

        x = self.conv3(x)

        x = self.res8(x)
        x = self.res9(x)
        x = self.res10(x)
        x = self.res11(x)
        x = self.res12(x)
        x = self.res13(x)

        x = self.conv4(x)

        x = self.res14(x)
        x = self.res15(x)
        x = self.res16(x)

        x = self.pool2(x)

        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return self.dense3(x)



@tf.function
def train_step(model, images, labels, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(model, images, labels, loss_object, test_loss, test_accuracy):
    predictions = model(images)

    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

def validate_model():
    global img_dir
    EPOCHS = 10
    BATCH_SIZE = 32
    model_path = os.path.join(BASE_DIR, 'Food_CNN/Training1/20210413-155855')

    model = tf.keras.models.load_model(model_path)
    model.summary()
   
    x_data = tfio.IODataset.from_hdf5("data_set1.hdf5", dataset="/x_data")
    y_data = tfio.IODataset.from_hdf5("data_set1.hdf5", dataset="/y_data")

    test_data = tf.data.Dataset.zip((x_data, y_data))
    test_data.shuffle(100009)

    test_data = test_data.batch(1, drop_remainder=True)

    loss_object = tf.keras.losses.CategoricalCrossentropy()
    test_loss_object = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
    for i, (img, label) in enumerate(test_data):
        predict = model(img)
        test_loss = loss_object(label, predict)
        test_loss_object(test_loss)
        test_accuracy(label, predict)
        print("predict = ", predict, "lbel = ", label, "loss = ", test_loss_object.result(), "acc = ", test_accuracy.result())

def convert_to_tfLite():
    
    food_dir = os.path.join(img_dir, "training")
    dir_list = os.listdir(food_dir)

    model_path = os.path.join(BASE_DIR, 'Food_CNN/Training1/20210501-184122')
    save_path = os.path.join(BASE_DIR, 'Food_CNN/tfLite/test_model2.tflite')
    label_path = os.path.join(BASE_DIR, 'Food_CNN/tfLite/test_label1.txt')
    
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

    tflite_model = converter.convert()

    labels = {}
    for idx, e in enumerate(dir_list):
        labels[e] = idx
    
    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    f.close()

    with open(label_path, 'w') as f:
        for idx, e in enumerate(dir_list):
            label = str(idx) + " " + e + "\n"
            print(label, end='')
            f.write(label)
    f.close()


# convert_to_tfLite()
# train_dir = os.path.join(img_dir,'training')
# test_dir = os.path.join(img_dir,'test')

# get_dataset(train_dir, test_dir)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, enable=True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs, ", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         print("---------------ERROR-----------------")
#         print(e)
#         print("---------------ERROR-----------------")

with tf.device('/GPU:0'):
    # keep_training()
    fit_and_save()
    # validate_model()
