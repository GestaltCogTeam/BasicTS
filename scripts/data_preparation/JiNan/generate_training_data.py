import json
import os
import shutil

import numpy as np
from generate_adj_mx import generate_adj_jinan as generate_adj

# Current path
current_dir = os.path.dirname(os.path.abspath(__file__))

# target path
base_dir = os.path.abspath(os.path.dirname(os.path.join(current_dir, '../..', '../..')))

# Hyperparameters
dataset_name = 'JiNan'
data_file_path = base_dir + f'/datasets/raw_data/{dataset_name}/{dataset_name}.npz'
distance_file_path = base_dir + f'/datasets/raw_data/{dataset_name}/{dataset_name}.csv'
graph_file_path = base_dir + f'/datasets/raw_data/{dataset_name}/adj_{dataset_name}.pkl'
'''data_file_path = f'/home/public/BasicTS_raw_data/{dataset_name}/{dataset_name}.npz'
distance_file_path = f'/home/public/BasicTS_raw_data/{dataset_name}/{dataset_name}.csv'
graph_file_path = f'/home/public/BasicTS_raw_data/{dataset_name}/adj_{dataset_name}.pkl'''
output_dir = base_dir + f'/datasets/{dataset_name}'
target_channel = [0]  # Target traffic flow channel
add_time_of_day = True  # Add time of day as a feature
add_day_of_week = True  # Add day of the week as a feature
add_day_of_month = False  # Add day of the month as a feature
add_day_of_year = False  # Add day of the year as a feature
steps_per_day = 288  # Number of time steps per day
frequency = 1440 // steps_per_day
domain = 'traffic flow'
timestamps_desc = ['time of day', 'day of week']
regular_settings = {
    'train_val_test_ratio': [0.6, 0.2, 0.2],
    'norm_each_channel': False,
    'rescale': True,
    'metrics': ['MAE', 'RMSE', 'MAPE'],
    'null_val': 0.0
}

def load_and_preprocess_data():
    '''Load and preprocess raw data, selecting the specified channel(s).'''
    data = np.load(data_file_path)['data']
    data = data[..., target_channel]
    if data.ndim == 3 and data.shape[-1] == 1:
        data = data.squeeze(axis=-1)
    print(f'Raw time series shape: {data.shape}')
    return data

def add_temporal_features(data):
    '''Add time of day and day of week as features to the data.'''
    l = data.shape[0]
    timestamps = []

    if add_time_of_day:
        time_of_day = np.array([i % steps_per_day / steps_per_day for i in range(l)])
        timestamps.append(time_of_day)

    if add_day_of_week:
        day_of_week = np.array([(i // steps_per_day) % 7 / 7 for i in range(l)])
        timestamps.append(day_of_week)

    timestamps = np.stack(timestamps, axis=-1)
    return timestamps

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
    np.save(os.path.join(output_dir, 'train_data.npy'), train_data)
    print(f"val_data shape: {val_data.shape}")
    np.save(os.path.join(output_dir, 'val_data.npy'), val_data)
    print(f"test_data shape: {test_data.shape}")
    np.save(os.path.join(output_dir, 'test_data.npy'), test_data)
    print(f"train_timestamps shape: {train_timestamps.shape}")
    np.save(os.path.join(output_dir, 'train_timestamps.npy'), train_timestamps)
    print(f"val_timestamps shape: {val_timestamps.shape}")
    np.save(os.path.join(output_dir, 'val_timestamps.npy'), val_timestamps)
    print(f"test_timestamps shape: {test_timestamps.shape}")
    np.save(os.path.join(output_dir, 'test_timestamps.npy'), test_timestamps)
    print(f'Data saved to {output_dir}')

def save_graph(df):
    n = df.shape[1]
    '''Save the adjacency matrix to the output directory, generating it if necessary.'''
    output_graph_path = os.path.join(output_dir, 'adj_mx.pkl')
    if os.path.exists(graph_file_path):
        shutil.copyfile(graph_file_path, output_graph_path)
    else:
        generate_adj(distance_file_path, graph_file_path, n)
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
    data = load_and_preprocess_data()

    # Add temporal features
    timestamps = add_temporal_features(data)

    # Save processed data
    split_and_save_data(data, timestamps)

    # Copy and save adjacency matrix
    save_graph(data)

    # Save dataset description
    save_description(data, timestamps)

if __name__ == '__main__':
    main()
