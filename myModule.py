import numpy as np
import os 
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import psutil
import threading

def read_csv(folder_path, type):
    """
    Reads all .csv files in the specified folder, extracts CPU or memory usage values,
    and returns a single flattened NumPy array of all valid float values, excluding missing or NaN data.

    Returns:
    numpy.ndarray: A 1D array of float values.
    """
    all_values = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith('.csv'):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                next(reader, None)  # Skip header

                for row in reader:
                    if len(row) < 2:
                        continue

                    if type == 'CPU':
                        value_str = row[1].strip().replace('%', '')
                    else:
                        mem_str = row[1].strip()
                        if 'GiB' in mem_str:
                            value_str = str(float(mem_str.replace(' GiB', '')) * 1024)  # Convert GiB to MiB
                        else:
                            value_str = mem_str.replace(' MiB', '')
                    
                    if value_str == '':
                        continue

                    try:
                        value = float(value_str)
                        if not np.isnan(value):
                            all_values.append(value)
                    except Exception:
                        continue  # Skip unparsable values

    return np.array(all_values)


# def read_csv(folder_path, T, type):
#     """
#     Reads all .csv files in the specified folder, extracts CPU stress percentage values,
#     and stores each time series as a numpy array inside a list, excluding missing or NaN data.

#     Returns:
#     list: A list of numpy arrays, each with shape (n,), where each value is a float (CPU usage)
#     """
#     time_series_list = []

#     for filename in sorted(os.listdir(folder_path)):
#         if filename.lower().endswith('.csv'):
#             file_path = os.path.join(folder_path, filename)
#             series = []

#             with open(file_path, newline='', encoding='utf-8') as csvfile:
#                 reader = csv.reader(csvfile, delimiter=',')
#                 next(reader, None)  # Skip header

#                 for row in reader:
#                     if len(row) < 2:
#                         continue

#                     if type == 'CPU':
#                         value_str = row[1].strip().replace('%', '')
#                     else:
#                         mem_str = row[1].strip()
#                         # value_str = mem_str.replace(' MiB', '').replace(' GiB', '')
#                         if 'GiB' in mem_str:
#                             value_str = str(float(mem_str.replace(' GiB', '')) * 1024)  # Convert GiB to MiB
#                         else:
#                             value_str = mem_str.replace(' MiB', '')
#                     if value_str == '':
#                         continue  # Skip empty values

#                     try:
#                         value = float(value_str)
#                         if np.isnan(value):  # or use np.isnan(value)
#                             continue  # Skip NaNs
#                         series.append(value)
#                     except Exception:
#                         continue  # Skip rows that can't be parsed

#             if len(series) > T:
#                 time_series_list.append(np.array(series[:T]))
#             # time_series_list.append(np.array(series[:]))
#     return np.array(time_series_list)



def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data)-seq_length-1):
        x.append(data[i:(i+seq_length)])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)



def model_size_in_bytes(model):
    total_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = 4  # usually 4 bytes for float32 parameters
    total_bytes = total_params * bytes_per_param
    return total_bytes

def pretty_size(bytes):
    # Helper to make bytes human-readable
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024
    return f"{bytes:.2f} TB"

















