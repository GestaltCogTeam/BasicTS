import os
import json
import pickle
import shutil

import numpy as np
import pandas as pd

# Hyperparameters
dataset_name = 'GBA'
data_file_path = f'datasets/raw_data/{dataset_name}/{dataset_name}.h5'
graph_file_path = f'datasets/raw_data/{dataset_name}/adj_{dataset_name}.npy'
meta_file_path = f'datasets/raw_data/{dataset_name}/meta_{dataset_name}.csv'
output_dir = f'datasets/{dataset_name}'
target_channel = [0]  # Target traffic flow channel
add_time_of_day = True  # Add time of day as a feature
add_day_of_week = True  # Add day of the week as a feature
add_day_of_month = False  # Add day of the month as a feature
add_day_of_year = False  # Add day of the year as a feature
steps_per_day = 96  # Number of time steps per day
frequency = 1440 // steps_per_day
domain = 'traffic flow'
feature_description = [domain, 'time of day', 'day of week']
regular_settings = {
    'INPUT_LEN': 12,
    'OUTPUT_LEN': 12,
    'TRAIN_VAL_TEST_RATIO': [0.6, 0.2, 0.2],
    'NORM_EACH_CHANNEL': False,
    'RESCALE': True,
    'METRICS': ['MAE', 'RMSE', 'MAPE'],
    'NULL_VAL': 0.0
}

def load_and_preprocess_data():
    '''Load and preprocess raw data, selecting the specified channel(s).'''
    df = pd.read_hdf(data_file_path)
    data = np.expand_dims(df.values, axis=-1)
    data = data[..., target_channel]
    print(f'Raw time series shape: {data.shape}')
    return data, df

def add_temporal_features(data, df):
    '''Add time of day and day of week as features to the data.'''
    _, n, _ = data.shape
    feature_list = [data]

    if add_time_of_day:
        time_of_day = (df.index.values - df.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
        time_of_day_tiled = np.tile(time_of_day, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(time_of_day_tiled)

    if add_day_of_week:
        day_of_week = df.index.dayofweek / 7
        day_of_week_tiled = np.tile(day_of_week, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(day_of_week_tiled)

    if add_day_of_month:
        # numerical day_of_month
        day_of_month = (df.index.day - 1 ) / 31 # df.index.day starts from 1. We need to minus 1 to make it start from 0.
        day_of_month_tiled = np.tile(day_of_month, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(day_of_month_tiled)

    if add_day_of_year:
        # numerical day_of_year
        day_of_year = (df.index.dayofyear - 1) / 366 # df.index.month starts from 1. We need to minus 1 to make it start from 0.
        day_of_year_tiled = np.tile(day_of_year, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(day_of_year_tiled)

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

def save_graph():
    '''Save the adjacency matrix to the output directory.'''
    output_graph_path = os.path.join(output_dir, 'adj_mx.pkl')
    adj_mx = np.load(graph_file_path)
    with open(output_dir + '/adj_mx.pkl', 'wb') as f:
        pickle.dump(adj_mx, f)
    print(f'Adjacency matrix saved to {output_graph_path}')

def save_meta_data():
    '''Save the meta data to the output directory'''
    output_meta_data_path = os.path.join(output_dir, 'meta.csv')
    shutil.copyfile(meta_file_path, output_meta_data_path)

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

    # Copy and save adjacency matrix
    save_graph()

    # Copy and save meta data
    save_meta_data()

    # Save dataset description
    save_description(data_with_features)

if __name__ == '__main__':
    main()
