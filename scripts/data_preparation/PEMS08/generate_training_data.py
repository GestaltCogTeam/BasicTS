import argparse
import pickle
import numpy as np
import os

"""
PEMS08 dataset (traffic flow dataset) default settings:
    - normalization:
        min-max norm
    - dataset division: 
        6:2:2
    - windows size:
        12
    - features:
        traffic flow
        --traffic occupy--(not used)
        --traffic speed--(not used)
        time in day
        day in week
    - target:
        predicting the traffic speed
"""

def MinMaxnormalization(data: np.array, output_dir: str, train_index: list) -> np.array:
    """min-max normalization.

    Args:
        data (np.array): raw time series data.
        output_dir (str): output dir path.
        train_index (list): train index.

    Returns:
        np.array: normalized raw time series data.
    """
    # L, N, C
    data_train = data[:train_index[-1][1], ...]

    _max = data_train.max(axis=(0, 1), keepdims=True)
    _min = data_train.min(axis=(0, 1), keepdims=True)

    print('max:', _max[0][0][0])
    print('min:', _min[0][0][0])
    scaler = {}
    scaler['func'] = re_max_min_normalization
    scaler['args'] = {"max":_max[0][0][0], "min":_min[0][0][0]}
    pickle.dump(scaler, open(output_dir + "/scaler.pkl", 'wb'))

    def normalize(x):
        x = 1. * (x - _min) / (_max - _min)
        x = 2. * x - 1.
        return x

    data_norm = normalize(data)

    return data_norm

def re_max_min_normalization(x, **kwargs):
    _min, _max = kwargs['min'][0, 0, 0], kwargs['max'][0, 0, 0]
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x

def generate_data(args):
    """preprocess and generate train/valid/test datasets.

    Args:
        args (Namespace): args for processing data.
    """
    C = args.C
    seq_len_short = args.seq_len_short
    add_time_in_day = True
    add_day_in_week = args.dow
    output_dir = args.output_dir

    # read data
    data = np.load(args.traffic_df_file_name)['data']
    data = data[..., C]
    print("Data shape: {0}".format(data.shape))

    L, N, F = data.shape
    num_samples_short = L - 2*seq_len_short + 1
    train_num_short = round(num_samples_short * train_ratio)
    valid_num_short = round(num_samples_short * valid_ratio)
    test_num_short  = num_samples_short - train_num_short - valid_num_short
    print("train_num_short:{0}".format(train_num_short))
    print("valid_num_short:{0}".format(valid_num_short))
    print("test_num_short:{0}".format(test_num_short))

    index_list      = []
    for i in range(seq_len_short, num_samples_short + seq_len_short):
        index = (i-seq_len_short, i, i+seq_len_short)
        index_list.append(index)
    train_index = index_list[:train_num_short]
    valid_index = index_list[train_num_short: train_num_short + valid_num_short]
    test_index  = index_list[train_num_short + valid_num_short: train_num_short + valid_num_short + test_num_short]
    
    scaler = MinMaxnormalization
    data_norm = scaler(data, output_dir, train_index)

    # add external feature
    feature_list = [data_norm]
    if add_time_in_day:
        # numerical time_in_day
        time_ind = [i%288 / 288 for i in range(data_norm.shape[0])]
        time_ind = np.array(time_ind)
        time_in_day = np.tile(time_ind, [1, N, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)
    if add_day_in_week:
        # numerical day_in_week
        day_in_week = [(i // 288)%7 for i in range(data_norm.shape[0])]
        day_in_week = np.array(day_in_week)
        day_in_week = np.tile(day_in_week, [1, N, 1]).transpose((2, 1, 0))
        feature_list.append(day_in_week)
    
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
    # seq_len_short   = 12
    train_ratio     = 0.6
    valid_ratio     = 0.2
    C               = [0]                   # selected channels

    name            = "PEMS08"
    dow             = True                  # if add day_of_week feature
    output_dir      = 'datasets/' + name
    data_file       = 'datasets/raw_data/{0}/{1}.npz'.format(name, name)
    
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
