import os
import json

import numpy as np
import pandas as pd

# Hyperparameters
dataset_name = 'Traffic'
data_file_path = f'datasets/raw_data/{dataset_name}/{dataset_name}.csv'
graph_file_path = None
output_dir = f'datasets/{dataset_name}'
target_channel = [0]  # Target traffic flow channel
add_time_of_day = True  # Add time of day as a feature
add_day_of_week = True  # Add day of the week as a feature
add_day_of_month = True  # Add day of the month as a feature
add_day_of_year = True  # Add day of the year as a feature
steps_per_day = 24  # Number of time steps per day
frequency = 1440 // steps_per_day
domain = 'road occupancy rates'
feature_description = [domain, 'time of day', 'day of week', 'day of week', 'day of year']
regular_settings = {
    'INPUT_LEN': 336,
    'OUTPUT_LEN': 336,
    'TRAIN_VAL_TEST_RATIO': [0.7, 0.1, 0.2],
    'NORM_EACH_CHANNEL': True,
    'RESCALE': False,
    'METRICS': ['MAE', 'MSE'],
    'NULL_VAL': np.nan
}

def load_and_preprocess_data():
    '''Load and preprocess raw data, selecting the specified channel(s).'''
    df = pd.read_csv(data_file_path)
    df_index = pd.to_datetime(df['date'].values, format='%Y-%m-%d %H:%M:%S').to_numpy()
    df = df[df.columns[1:]]
    df.index = df_index
    data = np.expand_dims(df.values, axis=-1)
    data = data[..., target_channel]
    print(f'Raw time series shape: {data.shape}')
    return data, df

def add_temporal_features(data, df):
    '''Add time of day and day of week as features to the data.'''
    _, n, _ = data.shape
    feature_list = [data]

    if add_time_of_day:
        # numerical time_of_day
        tod = (
            df.index.values - df.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
        tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)

    if add_day_of_week:
        # numerical day_of_week
        dow = df.index.dayofweek / 7
        dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    if add_day_of_month:
        # numerical day_of_month
        dom = (df.index.day - 1) / 31 # df.index.day starts from 1. We need to minus 1 to make it start from 0.
        dom_tiled = np.tile(dom, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dom_tiled)

    if add_day_of_year:
        # numerical day_of_year
        doy = (df.index.dayofyear - 1) / 366 # df.index.month starts from 1. We need to minus 1 to make it start from 0.
        doy_tiled = np.tile(doy, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(doy_tiled)

    data_with_features = np.concatenate(feature_list, axis=-1)  # L x N x C
    return data_with_features

def save_data(data):
    '''Save the preprocessed data to a binary file.'''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, 'data.dat')
    fp = np.memmap(file_path, dtype='float32', mode='w+', shape=data.shape)
    fp[:] = data[:]
    fp.flush()
    del fp
    print(f'Data saved to {file_path}')

def save_description(data):
    '''Save a description of the dataset to a JSON file.'''
    description = {
        'name': dataset_name,
        'domain': domain,
        'shape': data.shape,
        'num_time_steps': data.shape[0],
        'num_nodes': data.shape[1],
        'num_features': data.shape[2],
        'feature_description': feature_description,
        'has_graph': graph_file_path is not None,
        'frequency (minutes)': frequency,
        'regular_settings': regular_settings
    }
    description_path = os.path.join(output_dir, 'desc.json')
    with open(description_path, 'w') as f:
        json.dump(description, f, indent=4)
    print(f'Description saved to {description_path}')
    print(description)

def main():
    # Load and preprocess data
    data, df = load_and_preprocess_data()

    # Add temporal features
    data_with_features = add_temporal_features(data, df)

    # Save processed data
    save_data(data_with_features)

    # Save dataset description
    save_description(data_with_features)

if __name__ == '__main__':
    main()
