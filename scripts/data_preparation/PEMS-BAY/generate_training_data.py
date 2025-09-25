import os
import json
import shutil

import numpy as np
import pandas as pd


# Current path
current_dir = os.path.dirname(os.path.abspath(__file__))

# target path
base_dir = os.path.abspath(os.path.dirname(os.path.join(current_dir, '../..', '../..')))

# Hyperparameters
dataset_name = 'PEMS-BAY'
data_file_path = base_dir + f'/datasets/raw_data/{dataset_name}/{dataset_name}.h5'
graph_file_path = base_dir + f'/datasets/raw_data/{dataset_name}/adj_{dataset_name}.pkl'
output_dir = base_dir + f'/datasets/{dataset_name}'
target_channel = [0]  # Target traffic flow channel
add_time_of_day = True  # Add time of day as a feature
add_day_of_week = True  # Add day of the week as a feature
add_day_of_month = False  # Add day of the month as a feature
add_day_of_year = False  # Add day of the year as a feature
steps_per_day = 288  # Number of time steps per day
frequency = 1440 // steps_per_day
domain = 'traffic speed'
timestamps_desc = ['time of day', 'day of week']
regular_settings = {
    'train_val_test_ratio': [0.7, 0.1, 0.2],
    'norm_each_channel': False,
    'rescale': True,
    'metrics': ['MAE', 'RMSE', 'MAPE'],
    'null_val': 0.0
}

def load_and_preprocess_data():
    '''Load and preprocess raw data, selecting the specified channel(s).'''

    df = pd.read_hdf(data_file_path)
    print(f'Raw time series shape: {df.shape}')
    return df

def add_temporal_features(df):
    '''Add time of day and day of week as features to the data.'''
    timestamps = []

    if add_time_of_day:
        # numerical time_of_day
        tod = (df.index.values - df.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
        tod = np.array(tod)
        timestamps.append(tod)

    if add_day_of_week:
        # numerical day_of_week
        dow = df.index.dayofweek / 7
        timestamps.append(dow.values)

    if add_day_of_month:
        # numerical day_of_month
        dom = (df.index.day - 1) / 31 # df.index.day starts from 1. We need to minus 1 to make it start from 0.
        timestamps.append(dom.values)

    if add_day_of_year:
        # numerical day_of_year
        doy = (df.index.dayofyear - 1) / 366 # df.index.month starts from 1. We need to minus 1 to make it start from 0.
        timestamps.append(doy.values) # 李雨杰 debug 通过

    timestamps = np.stack(timestamps, axis=-1)
    return timestamps


def save_memmap(data, filename, dtype=np.float32):
    """使用 np.memmap 保存数据到磁盘"""
    # 创建内存映射文件（'w+' 表示读写模式）
    memmap = np.memmap(filename, dtype=dtype, shape=data.shape, mode='w+')
    # 将数据写入映射文件
    memmap[:] = data[:]
    # 刷新缓冲区，确保数据写入磁盘
    memmap.flush()
    # 删除引用，关闭文件
    del memmap


def split_and_save_data(data, timestamps):
    '''Save the preprocessed data to a binary file.'''
    train_ratio, val_ratio, _ = regular_settings['train_val_test_ratio']
    train_len = int(data.shape[0] * train_ratio)
    val_len = int(data.shape[0] * val_ratio)

    train_data = data[:train_len].astype(np.float32)
    val_data = data[train_len: train_len + val_len].astype(np.float32)
    test_data = data[train_len + val_len:].astype(np.float32)
    train_timestamps = timestamps[:train_len].astype(np.float32)
    val_timestamps = timestamps[train_len: train_len + val_len].astype(np.float32)
    test_timestamps = timestamps[train_len + val_len:].astype(np.float32)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"train_data shape: {train_data.shape}")
    save_memmap(train_data, os.path.join(output_dir, 'train_data.npy'))

    print(f"val_data shape: {val_data.shape}")
    save_memmap(val_data, os.path.join(output_dir, 'val_data.npy'))

    print(f"test_data shape: {test_data.shape}")
    save_memmap(test_data, os.path.join(output_dir, 'test_data.npy'))

    print(f"train_timestamps shape: {train_timestamps.shape}")
    save_memmap(train_timestamps, os.path.join(output_dir, 'train_timestamps.npy'))

    print(f"val_timestamps shape: {val_timestamps.shape}")
    save_memmap(val_timestamps, os.path.join(output_dir, 'val_timestamps.npy'))

    print(f"test_timestamps shape: {test_timestamps.shape}")
    save_memmap(test_timestamps, os.path.join(output_dir, 'test_timestamps.npy'))

    print(f'Data saved to {output_dir}')

def save_graph():
    '''Save the adjacency matrix to the output directory.'''
    output_graph_path = os.path.join(output_dir, 'adj_mx.pkl')
    shutil.copyfile(graph_file_path, output_graph_path)
    print(f'Adjacency matrix saved to {output_graph_path}')

def save_description(data, timestamps):
    '''Save a description of the dataset to a JSON file.'''
    description = {
        'name': dataset_name,
        'domain': domain,
        'frequency (minutes)': frequency,
        'shape': data.shape,
        'timestamps_shape': timestamps.shape,
        'timestamps_description': timestamps_desc,
        'num_time_steps': data.shape[0],
        'num_vars': data.shape[1],
        'has_graph': graph_file_path is not None,
        'regular_settings': regular_settings
    }
    description_path = os.path.join(output_dir, 'meta.json')
    with open(description_path, 'w') as f:
        json.dump(description, f, indent=4)
    print(f'Description saved to {description_path}')
    print(description)
    print('\n')

def main():
    print(f"---------- Generating {dataset_name} data ----------")

    # Load and preprocess data
    df = load_and_preprocess_data()

    # Add temporal features
    timestamps = add_temporal_features(df)

    # Save processed data
    split_and_save_data(df.values, timestamps)

    # Copy and save adjacency matrix
    save_graph()

    # Save dataset description
    save_description(df.values, timestamps)

if __name__ == '__main__':
    main()
