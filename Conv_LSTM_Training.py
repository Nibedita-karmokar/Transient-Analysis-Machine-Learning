#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:45:16 2024

@author: karmo005
"""


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import TimeDistributed, MaxPooling2D, Concatenate, Input, ConvLSTM2D, Conv2D, Conv2DTranspose, BatchNormalization, LayerNormalization
import numpy as np

import os
import re
import pandas as pd
import time

from tensorflow.keras.regularizers import l2


size_x = 128
size_y = 128
time_steps = 42  # Number of time steps

train_data_dir ="Training_data_dir"     #training data


def extract_numeric_part(filename, pattern):
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    return None


def sorted_files_by_numeric_parts(folder_path):
    file_names = os.listdir(folder_path)

    common_pattern = r"tile_data_(\d+)\.xlsx"

    numeric_parts = []
    for file_name in file_names:
        full_path = os.path.join(folder_path, file_name)
        numeric_part = extract_numeric_part(file_name, common_pattern)
        if numeric_part is not None:
            numeric_parts.append((numeric_part, full_path))

    # Sort the files based on the numeric parts
    sorted_files = [file_name for _, file_name in sorted(numeric_parts)]
    
    return sorted_files

def load_and_normalize_data_from_folder(folder_path):

    power_data = []
    temp_data = []
    
    file_list = sorted_files_by_numeric_parts(folder_path)
    
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        power_map, temp_map = read_image(file_path)
        power_data.append(power_map)
        temp_data.append(temp_map)

    # # Normalize the data using the updated global maximum values
    power_data = np.array(power_data).reshape(-1, time_steps, size_y, size_x) 
    temp_data = np.array(temp_data).reshape(-1, time_steps, size_y, size_x) 

    return power_data, temp_data



def read_image(fname):
    #df = pd.read_excel(fname)  
    df = pd.read_excel(fname, engine='openpyxl')  # Try 'openpyxl' engine

    power_map = np.zeros((size_y, size_x))
    temp_map = np.zeros((size_y, size_x))
     

    for index, row in df.iterrows():

        x = int(row['x'])
        y = int(row['y'])

        power_map[y,x] = row['P']
        temp_map[y,x] = row['T']
       

    return power_map,temp_map

def load_data_from_folder(folder_path):

    power_data = []
    temp_data = []
    
    file_list = sorted_files_by_numeric_parts(folder_path)
    
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        power_map, temp_map = read_image_vary(file_path)
        power_data.append(power_map)
        temp_data.append(temp_map)

    # Normalize the data using the updated global maximum values
    power_data = np.array(power_data)
    temp_data = np.array(temp_data)

    return power_data, temp_data



def read_image_vary(fname):
    df = pd.read_excel(fname, engine='openpyxl')

    # Get the last values of 'x' and 'y' columns
    last_x = df['x'].iloc[-1]
    last_y = df['y'].iloc[-1]

    # Determine size_x and size_y based on the last values
    size_x = last_x + 1  # Increment by 1 to account for 0-based indexing
    size_y = last_y + 1

    power_map = np.zeros((size_y, size_x))
    temp_map = np.zeros((size_y, size_x))

    for index, row in df.iterrows():
        x = int(row['x'])
        y = int(row['y'])

        power_map[y, x] = row['P']
        temp_map[y, x] = row['T']

    return power_map, temp_map

# pattern = "excel_files_new_k_dummy_\d+_\d+_\d+"  #new
pattern = "excel_files_new_\d+_\d+"    #original
    
train_folders = [folder for folder in os.listdir(train_data_dir) if re.match(pattern, folder)]

# # Load and normalize data from all training folders
P_data = []
T_data = []

for folder_name in train_folders:
    folder_path = os.path.join(train_data_dir, folder_name)
    
    # Load and normalize data for the current folder
    power_data, temp_data = load_and_normalize_data_from_folder(folder_path)
    P_data.extend(power_data)
    T_data.extend(temp_data)

P_data = np.array(P_data)
T_data = np.array(T_data)

print('P_data shape', P_data.shape, T_data.shape)


global_power_max = np.max(P_data)
global_temp_max = np.max(T_data)

max_temp_i = np.max(T_data[0])
max_power_i = np.max(P_data[0])

global_power_min = np.min(P_data)
global_temp_min = np.min(T_data)

P_data = (P_data-global_power_min)/(global_power_max-global_power_min)
T_data = (T_data-global_temp_min)/(global_temp_max-global_temp_min)


reg_rate =0.00005
def unet(input_shape, num_classes):
    inputs = Input(input_shape)

    conv0, pool0 = downsample_block(inputs, (3, 3), 32)
    conv1, pool1 = downsample_block(pool0, (3, 3), 64)
    conv2, pool2 = downsample_block(pool1, (3, 3), 128)
    conv3, pool3 = downsample_block(pool2, (3, 3), 256)
    conv4, pool4 = downsample_block(pool3, (3, 3), 512)
    convLSTM_encoded = ConvLSTM2D(filters=1024, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate), return_sequences=True)(pool4)

    up6 = upsample_block(convLSTM_encoded, conv4, (3, 3), 512)
    up7 = upsample_block(up6, conv3, (3, 3), 256)
    up8 = upsample_block(up7, conv2, (3, 3), 128)
    up9 = upsample_block(up8, conv1, (3, 3), 64)
    up10 = upsample_block(up9, conv0, (3, 3), 32)
    outputs = Conv2D(num_classes, (1, 1), activation='linear', padding='same')(up10)  # Linear activation for temperature map

    return Model(inputs=inputs, outputs=outputs)

def single_conv_block(input_layer, filter_size, filters):
    conv1 = TimeDistributed(Conv2D(filters, filter_size, activation='ReLU', padding='same', kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate)))(input_layer)

    return conv1


def downsample_block(x, filter_size, n_filters):
    f = single_conv_block(x, filter_size, n_filters)
    p = TimeDistributed(MaxPooling2D(2, padding='same'))(f)

    return f, p

# Define the upsample block function
def upsample_block(x, conv_features, filter_size, n_filters):
    x = TimeDistributed(Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding="same", kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate)))(x)

    x_shape = tf.shape(conv_features)
    x = tf.slice(x, tf.zeros(x_shape.shape, dtype=tf.dtypes.int32), x_shape)

    x = Concatenate(axis=4)([x, conv_features])
    x = TimeDistributed(Conv2D(n_filters, filter_size, activation='relu', padding='same', kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate)))(x)

    return x


height = None    # height can vary
width = None     # width can vary

channels = 1      
num_classes=1

input_shape = (time_steps, height, width, channels)

model = unet(input_shape, num_classes)

dataset = tf.data.Dataset.from_tensor_slices((P_data, T_data))

# Specify batch size and shuffle the dataset
batch_size = 1
dataset = dataset.shuffle(buffer_size=len(P_data)).batch(batch_size)
# Combine your data into a single dataset
combined_dataset = tf.data.Dataset.from_tensor_slices((P_data, T_data))

# Specify batch size and shuffle the combined dataset
buffer_size = len(P_data)  # Use the full buffer size for shuffling
combined_dataset = combined_dataset.shuffle(buffer_size=buffer_size).batch(batch_size)

# Define the split ratio
split_ratio = 0.85

# Calculate the number of samples for training and validation
num_samples = len(P_data)
num_train_samples = int(split_ratio * num_samples)
num_val_samples = num_samples - num_train_samples

# Split the dataset into training and validation
train_dataset = combined_dataset.take(num_train_samples)
val_dataset = combined_dataset.skip(num_train_samples)

# Extract data and labels from train_dataset
P_train = []
T_train = []
for data_batch, labels_batch in train_dataset:
    P_train.extend(data_batch.numpy())  # Assuming P_data and T_data are NumPy arrays
    T_train.extend(labels_batch.numpy())

# Extract data and labels from val_dataset
P_test = []
T_test = []
for data_batch, labels_batch in val_dataset:
    P_test.extend(data_batch.numpy())  # Assuming P_data and T_data are NumPy arrays
    T_test.extend(labels_batch.numpy())

# Convert the lists to NumPy arrays if needed
P_train = np.array(P_train)
T_train = np.array(T_train)
P_test = np.array(P_test)
T_test = np.array(T_test)

P_train_reshaped = P_train.reshape((-1, time_steps, size_y, size_x, 1))
T_train_reshaped = T_train.reshape((-1, time_steps, size_y, size_x, 1))
P_test_reshaped = P_test.reshape((-1, time_steps, size_y, size_x, 1))
T_test_reshaped = T_test.reshape((-1, time_steps, size_y, size_x, 1))

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=5,
    decay_rate=0.98,
    staircase=True)


start_time = time.time()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss='mse',
              metrics=['accuracy','mse', 'mae', 'mape'])


model.summary()

epochs = 200

# # Train your model
history = model.fit(
    x=P_train_reshaped,
    y=T_train_reshaped,
    validation_data=(P_test_reshaped, T_test_reshaped),
    epochs=epochs
)

# Print the model summary
end_time = time.time()

# Calculate runtime
runtime = end_time - start_time

# Print the runtime
print("Runtime:", runtime, "seconds")



