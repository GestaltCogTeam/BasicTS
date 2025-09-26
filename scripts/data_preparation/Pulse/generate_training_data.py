import json
import os

import numpy as np
import pandas as pd

# Current path
current_dir = os.path.dirname(os.path.abspath(__file__))

# target path
base_dir = os.path.abspath(os.path.dirname(os.path.join(current_dir, '../..', '../..')))

# Hyperparameters
dataset_name = 'Pulse'
data_file_path = base_dir + f'/datasets/raw_data/{dataset_name}/{dataset_name}.npy'
# data_file_path = f'/home/public/BasicTS_raw_data/{dataset_name}/{dataset_name}.npy'
graph_file_path = None
output_dir = base_dir + f'/datasets/{dataset_name}'
target_channel = [0]  
frequency = None
domain = 'simulated pulse data'
regular_settings = {
    'train_val_test_ratio': [0.7, 0.1, 0.2],
    # 'train_val_test_ratio': [0.0, 0.0, 1],
    'norm_each_channel': False,
    'rescale': True,
    'metrics': ['MAE', 'RMSE', 'MAPE'],
    'null_val': np.nan
}

def load_and_preprocess_data():
    '''Load and preprocess raw data, selecting the specified channel(s).'''
    data = np.load(data_file_path)
    data = data[..., target_channel]
    
    if data.ndim == 3 and data.shape[-1] == 1:
        data = data.squeeze(axis=-1)
    
    print(f'Raw time series shape: {data.shape}')
    return data

def split_and_save_data(data):
    '''Save the preprocessed data to a binary file.'''
    train_ratio, val_ratio, _ = regular_settings['train_val_test_ratio']
    train_len = int(data.shape[0] * train_ratio)
    val_len = int(data.shape[0] * val_ratio)

    train_data = data[:train_len].astype(np.float32)
    val_data = data[train_len: train_len + val_len].astype(np.float32)
    test_data = data[train_len + val_len:].astype(np.float32)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"train_data shape: {train_data.shape}")
    np.save(os.path.join(output_dir, 'train_data.npy'), train_data)
    print(f"val_data shape: {val_data.shape}")
    np.save(os.path.join(output_dir, 'val_data.npy'), val_data)
    print(f"test_data shape: {test_data.shape}")
    np.save(os.path.join(output_dir, 'test_data.npy'), test_data)

    print(f'Data saved to {output_dir}')


def save_description(data):
    '''Save a description of the dataset to a JSON file.'''
    description = {
        'name': dataset_name,
        'domain': domain,
        'frequency (minutes)': frequency,
        'shape': data.shape,
        'timestamps_shape': None,
        'timestamps_description': None,
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
    data = load_and_preprocess_data()

    # Save processed data
    split_and_save_data(data)

    # Save dataset description
    save_description(data)

if __name__ == '__main__':
    main()
