import json
import os

import numpy as np
import pandas as pd

# Current path
current_dir = os.path.dirname(os.path.abspath(__file__))

# target path
base_dir = os.path.abspath(os.path.dirname(os.path.join(current_dir, '../..', '../..')))

# Hyperparameters
dataset_name = 'Illness'
# data_file_path = f'datasets/BasicTS_raw_data/{dataset_name}/{dataset_name}.csv'
data_file_path = base_dir + f'/datasets/raw_data/{dataset_name}/{dataset_name}.csv'
graph_file_path = None
output_dir = base_dir + f'/datasets/{dataset_name}' 
target_channel = [0]  # Target traffic flow channel 
add_time_of_day = True  # Add time of day as a feature
add_day_of_week = True  # Add day of the week as a feature
add_day_of_month = True  # Add day of the month as a feature
add_day_of_year = True  # Add day of the year as a feature
steps_per_day = 1/7  # Number of time steps per day
frequency = 1440 // steps_per_day
domain = 'ilness data'
timestamps_desc = ['time of day', 'day of week', 'day of month', 'day of year']
regular_settings = {
    'train_val_test_ratio': [0.7, 0.1, 0.2],
    # 'train_val_test_ratio': [0.0, 0.0, 1], 
    'norm_each_channel': True,
    'rescale': False,
    'metrics': ['MAE', 'MSE'],
    'null_val': np.nan
}

def load_and_preprocess_data():
    '''Load and preprocess raw data, selecting the specified channel(s).'''

    df = pd.read_csv(data_file_path)
    df_index = pd.to_datetime(df['date'].values, format='%Y-%m-%d %H:%M:%S').to_numpy()
    df = df[df.columns[1:]]
    df.index = df_index
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
        dom = (df.index.day - 1) / 31  # df.index.day starts from 1. We need to minus 1 to make it start from 0.
        timestamps.append(dom.values)

    if add_day_of_year:
        # numerical day_of_year
        doy = (df.index.dayofyear - 1) / 366  # df.index.month starts from 1. We need to minus 1 to make it start from 0.
        timestamps.append(doy.values)

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
    print('\n')

def main():
    print(f"---------- Generating {dataset_name} data ----------")

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
