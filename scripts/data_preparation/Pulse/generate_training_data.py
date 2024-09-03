import os
import json

import numpy as np


# Hyperparameters
dataset_name = 'Pulse'
data_file_path = f'datasets/raw_data/{dataset_name}/{dataset_name}.npy'
graph_file_path = None
output_dir = f'datasets/{dataset_name}'
target_channel = [0]  # Target traffic flow channel
frequency = None
domain = 'simulated pulse data'
feature_description = [domain]
regular_settings = {
    'INPUT_LEN': 336,
    'OUTPUT_LEN': 336,
    'TRAIN_VAL_TEST_RATIO': [0.7, 0.1, 0.2],
    'NORM_EACH_CHANNEL': False,
    'RESCALE': True,
    'METRICS': ['MAE', 'RMSE', 'MAPE'],
    'NULL_VAL': np.nan
}

def load_and_preprocess_data():
    '''Load and preprocess raw data, selecting the specified channel(s).'''
    data = np.load(data_file_path)
    data = data[..., target_channel]
    print(f'Raw time series shape: {data.shape}')
    return data

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
        'settings': regular_settings
    }
    description_path = os.path.join(output_dir, 'desc.json')
    with open(description_path, 'w') as f:
        json.dump(description, f, indent=4)
    print(f'Description saved to {description_path}')
    print(description)

def main():
    # Load and preprocess data
    data = load_and_preprocess_data()

    # Save processed data
    save_data(data)

    # Save dataset description
    save_description(data)

if __name__ == '__main__':
    main()
