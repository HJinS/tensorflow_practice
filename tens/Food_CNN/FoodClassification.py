import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil, os
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import HDF5Matrix, normalize
import tensorflow as tf
import tensorflow_io as tfio
import h5py

BASE_DIR = os.path.abspath('..')

img_dir = os.path.join(BASE_DIR,"Food_data\dataset")

X_data, Y_data = [], []
categories = []

def get_dataset(img_dir):
    dir_list = os.listdir(img_dir)
    file = h5py.File('data_set.hdf5', 'w')
    x_dataset = file.create_dataset("x_data", (100009, 250, 250, 3), dtype='float32')
    y_dataset = file.create_dataset("y_data", (100009, 100), dtype='float32')
    idx = 0
    gif_cnt = 0
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
            img = image.load_img(fin_loc, target_size=(250, 250))
            img_tensor = image.img_to_array(img)
            img_tensor = img_tensor / 255.0
            label_classes = [0] * 100
            label_classes[len(categories)-1] = 1
            x_dataset[idx] = img_tensor
            y_dataset[idx] = label_classes
            idx += 1
    print("idx = ", idx)
    print("git_cnt = ", gif_cnt)


# get_dataset(img_dir)
BATCH_SIZE = 2048

working_dir = os.path.join(BASE_DIR,"Food_CNN")
file_path = os.path.join(working_dir,"data_set.hdf5")

x_data = tfio.IODataset.from_hdf5("data_set.hdf5", dataset="/x_data")
y_data = tfio.IODataset.from_hdf5("data_set.hdf5", dataset="/y_data")

train = tf.data.Dataset.zip((x_data, y_data)).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

print(train)




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