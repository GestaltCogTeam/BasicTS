import argparse
import pickle
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(__file__ + "/../../../.."))
from basicts.data.transform import standard_transform


"""
Solar Energy dataset (Energy-related dataset) default settings:
    - source:
        multivariate-time-series-data: https://github.com/laiguokun/multivariate-time-series-data
    - normalization:
        standard norm
    - dataset division:
        6:2:2
    - windows size:
        12
"""
"""
cp /home/wjk/miniconda3/pkgs/libstdcxx-ng-9.3.0-hd4cf53a_17/lib/libstdc++.so.6.0.28

"""

def generate_data(args):
    """preprocess and generate train/valid/test datasets.

    Args:
        args (Namespace): args for processing data.
    """
    C = args.C
    future_seq_len  = args.future_seq_len
    history_seq_len = args.history_seq_len
    add_time_in_day = True
    add_day_in_week = args.dow
    output_dir = args.output_dir

    # read data
    data = np.loadtxt(args.data_file_path, delimiter=',')
    data = np.expand_dims(data, axis=-1)
    data = data[..., C]
    print("Data shape: {0}".format(data.shape))

    L, N, F = data.shape
    num_samples     = L - (history_seq_len + future_seq_len) + 1
    train_num_short = round(num_samples * train_ratio)
    valid_num_short = round(num_samples * valid_ratio)
    test_num_short  = num_samples - train_num_short - valid_num_short
    print("train_num_short:{0}".format(train_num_short))
    print("valid_num_short:{0}".format(valid_num_short))
    print("test_num_short:{0}".format(test_num_short))

    index_list      = []
    for t in range(history_seq_len, num_samples + history_seq_len):
        index = (t-history_seq_len, t, t+future_seq_len)
        index_list.append(index)
    train_index = index_list[:train_num_short]
    valid_index = index_list[train_num_short: train_num_short + valid_num_short]
    test_index  = index_list[train_num_short + valid_num_short: train_num_short + valid_num_short + test_num_short]
    
    scaler = standard_transform
    data_norm = scaler(data, output_dir, train_index, history_seq_len, future_seq_len)

    # add external feature
    feature_list = [data_norm]
    if add_time_in_day:
        # numerical time_in_day
        time_ind    = [i%args.steps_per_day / args.steps_per_day for i in range(data_norm.shape[0])]
        time_ind    = np.array(time_ind)
        time_in_day = np.tile(time_ind, [1, N, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)
    if add_day_in_week:
        # numerical day_in_week
        day_in_week = [(i // args.steps_per_day)%7 for i in range(data_norm.shape[0])]
        day_in_week = np.array(day_in_week)
        day_in_week = np.tile(day_in_week, [1, N, 1]).transpose((2, 1, 0))
        feature_list.append(day_in_week)

    processed_data = np.concatenate(feature_list, axis=-1)

    # dump data
    index = {}
    index['train'] = train_index
    index['valid'] = valid_index
    index['test']  = test_index
    pickle.dump(index, open(output_dir + "/index_in{}_out{}.pkl".format(history_seq_len, future_seq_len), "wb"))

    data = {}
    data['processed_data'] = processed_data
    pickle.dump(data, open(output_dir + "/data_in{0}_out{1}.pkl".format(history_seq_len, future_seq_len), "wb"))

if __name__ == "__main__":
    history_seq_len = 168                    # sliding window size for generating history sequence and target sequence
    future_seq_len  = 12
    # print("future_squence_len: {}".format(future_seq_len))
    train_ratio     = 0.6
    valid_ratio     = 0.2
    C               = [0]                   # selected channels
    steps_per_day   = 4                     # 6 hours
    # print("start!")
    name            = "Beijing_air_quality"
    dow             = True                  # if add day_of_week feature
    output_dir      = 'datasets/' + name
    data_file_path  = 'datasets/raw_data/{0}/{1}.csv'.format(name, name)
    # print(data_file_path)
    
    parser  = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=output_dir, help="Output directory.")
    parser.add_argument("--data_file_path", type=str, default=data_file_path, help="Raw traffic readings.")
    parser.add_argument("--history_seq_len", type=int, default=history_seq_len, help="Sequence Length.")
    parser.add_argument("--future_seq_len", type=int, default=future_seq_len, help="Sequence Length.")
    parser.add_argument("--steps_per_day", type=int, default=steps_per_day, help="Sequence Length.")
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
