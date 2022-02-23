import argparse
import pickle
import numpy as np
import os
import pandas as pd

"""
METR-LA dataset (traffic speed dataset) default settings:
    - normalization:
        standard norm
    - dataset division: 
        7:1:2
    - windows size:
        12
    - features:
        traffic speed
        time in day
        day in week
    - target:
        predicting the traffic speed
"""

def standard_transform(data: np.array, output_dir: str, train_index: list) -> np.array:
    """standard normalization.

    Args:
        data (np.array): raw time series data.
        output_dir (str): output dir path.
        train_index (list): train index.

    Returns:
        np.array: normalized raw time series data.
    """
    # data: L, N, C
    data_train  = data[:train_index[-1][1], ...]
    
    mean, std   = data_train[..., 0].mean(), data_train[..., 0].std()

    print("mean (training data):", mean)
    print("std (training data):", std)
    scaler = {}
    scaler['func'] = standard_re_transform.__name__
    scaler['args'] = {"mean":mean, "std":std}
    pickle.dump(scaler, open(output_dir + "/scaler.pkl", 'wb'))
    
    def normalize(x):
        return (x - mean) / std
    
    data_norm = normalize(data)
    return data_norm

def standard_re_transform(x, **kwargs):
    mean, std = kwargs['mean'], kwargs['std']
    x = x * std
    x = x + mean
    return x

def generate_data(args):
    """preprocess and generate train/valid/test datasets.

    Args:
        args (Namespace): args for processing data.
    """
    C = args.C
    seq_len_short   = args.seq_len_short
    add_time_in_day = True
    add_day_in_week = args.dow
    output_dir      = args.output_dir

    # read data
    df   = pd.read_hdf(args.traffic_df_file_name)
    data = np.expand_dims(df.values, axis=-1)

    data = data[..., C]
    print("Data shape: {0}".format(data.shape))

    L, N, F = data.shape
    num_samples_short   = L - 2*seq_len_short + 1
    train_num_short     = round(num_samples_short * train_ratio)
    valid_num_short     = round(num_samples_short * valid_ratio)
    test_num_short      = num_samples_short - train_num_short - valid_num_short
    print("train_num_short:{0}".format(train_num_short))
    print("valid_num_short:{0}".format(valid_num_short))
    print("test_num_short:{0}".format(test_num_short))

    index_list = []
    for i in range(seq_len_short, num_samples_short + seq_len_short):
        index = (i-seq_len_short, i, i+seq_len_short)
        index_list.append(index)

    train_index = index_list[:train_num_short]
    valid_index = index_list[train_num_short: train_num_short + valid_num_short]
    test_index  = index_list[train_num_short + valid_num_short: train_num_short + valid_num_short + test_num_short]
    
    scaler      = standard_transform
    data_norm   = scaler(data, output_dir, train_index)

    # add external feature
    feature_list = [data_norm]
    if add_time_in_day:
        # numerical time_in_day
        time_ind    = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, N, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)

    if add_day_in_week:
        # numerical day_in_week
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, N, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    raw_data = np.concatenate(feature_list, axis=-1)

    # dump data
    index = {}
    index['train'] = train_index
    index['valid'] = valid_index
    index['test']  = test_index
    pickle.dump(index, open(output_dir + "/index.pkl", "wb"))

    data = {}
    data['raw_data'] = raw_data
    pickle.dump(data, open(output_dir + "/data.pkl", "wb"))

if __name__ == "__main__":
    window_size     = 12                    # sliding window size for generating history sequence and target sequence
    train_ratio     = 0.7
    valid_ratio     = 0.1
    C               = [0]                   # selected channels

    name            = "METR-LA"
    dow             = True                  # if add day_of_week feature
    output_dir      = 'datasets/' + name
    data_file       = 'datasets/raw_data/{0}/{1}.h5'.format(name, name)
    
    parser  = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=output_dir, help="Output directory.")
    parser.add_argument("--traffic_df_file_name", type=str, default=data_file, help="Raw traffic readings.",)
    parser.add_argument("--seq_len_short", type=int, default=window_size, help="Sequence Length.",)
    parser.add_argument("--dow", type=bool, default=dow, help='Add feature day_of_week.')
    parser.add_argument("--C", type=list, default=C, help='Selected channels.')
    parser.add_argument("--train_ratio", type=float, default=train_ratio, help='Train ratio')
    parser.add_argument("--valid_ratio", type=float, default=valid_ratio, help='Validate ratio.')
    
    args    = parser.parse_args()
    if os.path.exists(args.output_dir):
        reply   = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y': exit
    else:
        os.makedirs(args.output_dir)
    generate_data(args)
