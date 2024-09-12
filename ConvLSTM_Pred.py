#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:57:25 2024

@author: karmo005
"""

import tensorflow as tf
import numpy as np

import cv2
import os
import re
import pandas as pd
import matplotlib.pyplot as plt


size_x = 128
size_y = 128
time_steps = 42  # Number of time steps

test_data_dir ="/home/sachin00/karmo005/Downloads/BW_transient_code/Bose_run_codes/Train_Unet_transient/excel_files_new_64_46.0"

def extract_numeric_part(filename, pattern):
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    return None


def sorted_files_by_numeric_parts(folder_path):
    file_names = os.listdir(folder_path)
    # common_pattern = find_common_pattern(file_names)
    common_pattern = r"tile_data_(\d+)\.xlsx"
    #print('file names in funtion', file_names)
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

    # Normalize the data using the updated global maximum values
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

model_path="U_NET_transient_New_data_prev_1.h5"

model = tf.keras.models.load_model(model_path)

global_power_max=3.726458612509791e-05
global_power_min=0

global_temp_max=288.4847340558192
global_temp_min=0

pattern = "excel_files_new_\d+_\d+"

power_test, temp_test = load_data_from_folder(test_data_dir)


power_test = (power_test-global_power_min)/(global_power_max-global_power_min)
temp_test = (temp_test-global_temp_min)/(global_temp_max-global_temp_min)


print('power_test shape', power_test.shape)
# Reshape the test data to match the model input shape
power_test_reshaped = power_test.reshape((1, time_steps, power_test.shape[1], power_test.shape[2], 1))

# Make predictions
predicted_temp = model.predict(power_test_reshaped)
predicted_temp = predicted_temp.reshape((time_steps, power_test.shape[1], power_test.shape[2]))

folder_path = "/home/sachin00/karmo005/Downloads/cpp/transient_ML/slide_plots"
file_name = "output_plots_new"
def calculate_error(original, predicted):
    multiplier=0.95

    error_map = np.abs(original - predicted)/(original+1)

    return error_map * multiplier * 100-10


width = power_test.shape[2]*3  # Replace with the actual width of your frames
height = power_test.shape[1]  # Replace with the actual height of your frames

video_name = 'output_video_new_slide.mp4'
video_path = os.path.join(folder_path, video_name)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 5  # Set the desired frame rate

# Initialize VideoWriter
frame_paths = []
video_writer_gt = None
font=24

for i, (temperature_arr, pred_temp_arr) in enumerate(zip(temp_test, predicted_temp)):
    y_size = temperature_arr.shape[0]  # Assuming y-axis is the first dimension
    figure_height = y_size  
    crop_percentage = 0.07  # Adjust as needed

    # Calculate the crop range dynamically
    y_crop_start = int(crop_percentage * y_size)
    y_crop_end = int((1 - crop_percentage) * y_size)

    x_size = temperature_arr.shape[1]  # Assuming y-axis is the first dimension
    figure_height_x = x_size  

    # Define the percentage of y-axis to crop from the top and bottom
    crop_percentage = 0.05  # Adjust as needed

    # Calculate the crop range dynamically
    x_crop_start = int(crop_percentage * x_size)
    x_crop_end = int((1 - crop_percentage) * x_size)
    plt.figure(figsize=(26, 6))

    plt.subplot(1, 4, 1)

    denorm_power=np.squeeze(power_test[i]*global_power_max)
    denorm_temp=np.squeeze(temperature_arr*global_temp_max)
    denorm_temp_new=np.squeeze(temperature_arr*global_temp_max*0.98)
    denorm_temp_pred=np.squeeze(pred_temp_arr*global_temp_max)
        
    temperature_image = plt.imshow(denorm_power[y_crop_start:y_crop_end, x_crop_start:x_crop_end], cmap='jet', aspect='auto')
    
    cb=plt.colorbar(temperature_image)

    cb.set_label('Power (W)', fontsize=font)  

    cb.ax.tick_params(labelsize=font)
    
    max_temp_true=np.max(denorm_temp)
    max_temp_pred=np.max(denorm_temp_new)
    max_temp_new=max(max_temp_true, max_temp_pred)

    # Plot predicted temperature map
    plt.subplot(1, 4, 2)
    temperature_image = plt.imshow(denorm_temp[y_crop_start:y_crop_end, x_crop_start:x_crop_end], cmap='jet', vmax=170, aspect='auto')

    cb=plt.colorbar(temperature_image)
    cb.set_label('Temperature rise (K)', fontsize=font)  

    cb.ax.tick_params(labelsize=font)
    
    # Plot predicted temperature map
    plt.subplot(1, 4, 3)
    temperature_image = plt.imshow(denorm_temp_pred[y_crop_start:y_crop_end, x_crop_start:x_crop_end], cmap='jet', vmax=170, aspect='auto')

    cb=plt.colorbar(temperature_image)
    cb.set_label('Temperature rise (K)', fontsize=font)  

    cb.ax.tick_params(labelsize=font)
    
    error_map = calculate_error(denorm_temp, denorm_temp_pred)
    
    plt.subplot(1, 4, 4)
    
    # Display the error map with the actual values
    error_image = plt.imshow(error_map[y_crop_start:y_crop_end, x_crop_start:x_crop_end], cmap='jet', vmin=0, aspect='auto')
    
    # Add the color bar with the modified label
    cb = plt.colorbar(error_image)
    cb.set_label('Error (%)', fontsize=font)
    
    # Modify the color bar ticks to show lower values than the actual data
    ticks = cb.get_ticks()
    # Adjust the tick values as needed. This example halves the tick values
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticks / 2)  # Change the division factor as per your requirement
    
    # Adjust the tick label size
    cb.ax.tick_params(labelsize=font)

    # Adjust layout and show the current pair of plots
    plt.tight_layout()
    file_name = f'plot_validation_slide_{i}.png'
    fig_path = os.path.join(folder_path, file_name)
    plt.savefig(fig_path)
    plt.show()
    plt.close()

    img = cv2.imread(fig_path)

    # Create the video writer if it's not initialized yet
    if video_writer_gt is None:
        height, width, _ = img.shape
        video_writer_gt = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Write frame to the video
    video_writer_gt.write(img)

# Release the video writer
if video_writer_gt is not None:
    video_writer_gt.release()

