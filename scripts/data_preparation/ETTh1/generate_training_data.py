import json
import os

import numpy as np
import pandas as pd

# Hyperparameters
dataset_name = 'ETTh1'
data_file_path = f'datasets/BasicTS_raw_data/{dataset_name}/{dataset_name}.csv'
graph_file_path = None
output_dir = f'datasets/{dataset_name}'
target_channel = [0]  # Target traffic flow channel
add_time_of_day = True  # Add time of day as a feature
add_day_of_week = True  # Add day of the week as a feature
add_day_of_month = True  # Add day of the month as a feature
add_day_of_year = True  # Add day of the year as a feature
steps_per_day = 24  # Number of time steps per day
frequency = 1440 // steps_per_day
domain = 'electricity transformer temperature'
timestamps_desc = ['time of day', 'day of week', 'day of month', 'day of year']
regular_settings = {
    'train_val_test_ratio': [0.6, 0.2, 0.2],
    'norm_each_channel': True,
    'rescale': False,
    'metrics': ['MAE', 'MSE'],
    'null_val': np.nan
}

def load_and_preprocess_data():
    '''Load and preprocess raw data, selecting the specified channel(s).'''
    df = pd.read_csv(data_file_path)
    df = df.iloc[:20*30*24]
    df_index = pd.to_datetime(df['date'].values, format='%Y-%m-%d %H:%M:%S').to_numpy()
    df = df[df.columns[1:]]
    df.index = df_index
    print(f'Raw time series shape: {df.shape}')
    return df

def add_temporal_features(df):
    '''Add time of day and day of week as features to the data.'''
    l = df.shape[0]
    timestamps = []

    if add_time_of_day:
        # numerical time_of_day
        tod = np.array([i % steps_per_day / steps_per_day for i in range(l)])
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
        timestamps.append(dom.values)

    timestamps = np.stack(timestamps, axis=-1)
    return timestamps

def split_and_save_data(data, timestamps):
    '''Save the preprocessed data to a binary file.'''
    train_ratio, val_ratio, _ = regular_settings['train_val_test_ratio']
    train_len = int(data.shape[0] * train_ratio)
    val_len = int(data.shape[0] * val_ratio)
    
    train_data = data[:train_len].astype(np.float32)
    val_data = data[train_len : train_len + val_len].astype(np.float32)
    test_data = data[train_len + val_len :].astype(np.float32)
    train_timestamps = timestamps[:train_len].astype(np.float32)
    val_timestamps = timestamps[train_len : train_len + val_len].astype(np.float32)
    test_timestamps = timestamps[train_len + val_len :].astype(np.float32)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(os.path.join(output_dir, 'train_data.npy'), train_data)
    np.save(os.path.join(output_dir, 'val_data.npy'), val_data)
    np.save(os.path.join(output_dir, 'test_data.npy'), test_data)
    np.save(os.path.join(output_dir, 'train_timestamps.npy'), train_timestamps)
    np.save(os.path.join(output_dir, 'val_timestamps.npy'), val_timestamps)
    np.save(os.path.join(output_dir, 'test_timestamps.npy'), test_timestamps)
    print(f'Data saved to {output_dir}')

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
        'regular_settings': regular_settings,
    }
    description_path = os.path.join(output_dir, 'meta.json')
    with open(description_path, 'w') as f:
        json.dump(description, f, indent=4)
    print(f'Description saved to {description_path}')
    print(description)

def main():
    # Load and preprocess data
    df = load_and_preprocess_data()

    # Add temporal features
    timestamps = add_temporal_features(df)

    # Save processed data
    split_and_save_data(df.values, timestamps)

    # Save dataset description
    save_description(df.values, timestamps)

if __name__ == '__main__':
    main()
