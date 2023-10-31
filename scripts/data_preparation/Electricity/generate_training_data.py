import os
import sys
import pickle
import argparse

import numpy as np
import pandas as pd

# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../../.."))
from basicts.data.transform import standard_transform


def generate_data(args: argparse.Namespace):
    """Preprocess and generate train/valid/test datasets.

    Args:
        args (argparse): configurations of preprocessing
    """

    target_channel = args.target_channel
    future_seq_len = args.future_seq_len
    history_seq_len = args.history_seq_len
    add_time_of_day = args.tod
    add_day_of_week = args.dow
    add_day_of_month = args.dom
    add_day_of_year = args.doy
    output_dir = args.output_dir
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    data_file_path = args.data_file_path
    steps_per_day = args.steps_per_day
    norm_each_channel = args.norm_each_channel
    if_rescale = not norm_each_channel # if evaluate on rescaled data. see `basicts.runner.base_tsf_runner.BaseTimeSeriesForecastingRunner.build_train_dataset` for details.

    # read data
    df = pd.read_csv(data_file_path)
    df_index = pd.to_datetime(df["date"].values, format="%Y-%m-%d %H:%M").to_numpy()
    df = df[df.columns[1:]]
    df.index = df_index

    data = np.expand_dims(df.values, axis=-1)

    data = data[..., target_channel]
    print("raw time series shape: {0}".format(data.shape))

    # split data
    l, n, f = data.shape
    num_samples = l - (history_seq_len + future_seq_len) + 1
    train_num = round(num_samples * train_ratio)
    valid_num = round(num_samples * valid_ratio)
    test_num = num_samples - train_num - valid_num
    print("number of training samples:{0}".format(train_num))
    print("number of validation samples:{0}".format(valid_num))
    print("number of test samples:{0}".format(test_num))

    index_list = []
    for t in range(history_seq_len, num_samples + history_seq_len):
        index = (t-history_seq_len, t, t+future_seq_len)
        index_list.append(index)

    train_index = index_list[:train_num]
    valid_index = index_list[train_num: train_num + valid_num]
    test_index = index_list[train_num +
                            valid_num: train_num + valid_num + test_num]

    # normalize data
    scaler = standard_transform
    # Following related works (e.g. informer and autoformer), we normalize each channel separately.
    data_norm = scaler(data, output_dir, train_index, history_seq_len, future_seq_len, norm_each_channel=norm_each_channel)

    # add temporal feature
    feature_list = [data_norm]
    if add_time_of_day:
        # numerical time_of_day
        tod = [i % steps_per_day / steps_per_day for i in range(data_norm.shape[0])]
        tod = np.array(tod)
        tod_tiled = np.tile(tod, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(tod_tiled)

    if add_day_of_week:
        # numerical day_of_week
        dow = df.index.dayofweek / 7
        dow_tiled = np.tile(dow, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    if add_day_of_month:
        # numerical day_of_month
        dom = (df.index.day - 1 ) / 31 # df.index.day starts from 1. We need to minus 1 to make it start from 0.
        dom_tiled = np.tile(dom, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(dom_tiled)

    if add_day_of_year:
        # numerical day_of_year
        doy = (df.index.dayofyear - 1) / 366 # df.index.month starts from 1. We need to minus 1 to make it start from 0.
        doy_tiled = np.tile(doy, [1, n, 1]).transpose((2, 1, 0))
        feature_list.append(doy_tiled)

    processed_data = np.concatenate(feature_list, axis=-1)

    # save data
    index = {}
    index["train"] = train_index
    index["valid"] = valid_index
    index["test"] = test_index
    with open(output_dir + "/index_in_{0}_out_{1}_rescale_{2}.pkl".format(history_seq_len, future_seq_len, if_rescale), "wb") as f:
        pickle.dump(index, f)

    data = {}
    data["processed_data"] = processed_data
    with open(output_dir + "/data_in_{0}_out_{1}_rescale_{2}.pkl".format(history_seq_len, future_seq_len, if_rescale), "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    # sliding window size for generating history sequence and target sequence
    HISTORY_SEQ_LEN = 168
    FUTURE_SEQ_LEN = 96

    TRAIN_RATIO = 0.7
    VALID_RATIO = 0.1
    TARGET_CHANNEL = [0]                   # target channel(s)
    STEPS_PER_DAY = 24          # every 1 hour

    DATASET_NAME = "Electricity"      # sampling frequency: every 1 hour
    TOD = True                  # if add time_of_day feature
    DOW = True                  # if add day_of_week feature
    DOM = True                  # if add day_of_month feature
    DOY = True                  # if add day_of_year feature

    OUTPUT_DIR = "datasets/" + DATASET_NAME
    DATA_FILE_PATH = "datasets/raw_data/{0}/{0}.csv".format(DATASET_NAME)

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--data_file_path", type=str,
                        default=DATA_FILE_PATH, help="Raw traffic readings.")
    parser.add_argument("--history_seq_len", type=int,
                        default=HISTORY_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--future_seq_len", type=int,
                        default=FUTURE_SEQ_LEN, help="Sequence Length.")
    parser.add_argument("--steps_per_day", type=int,
                        default=STEPS_PER_DAY, help="Sequence Length.")
    parser.add_argument("--tod", type=bool, default=TOD,
                        help="Add feature time_of_day.")
    parser.add_argument("--dow", type=bool, default=DOW,
                        help="Add feature day_of_week.")
    parser.add_argument("--dom", type=bool, default=DOM,
                        help="Add feature day_of_week.")
    parser.add_argument("--doy", type=bool, default=DOY,
                        help="Add feature day_of_week.")
    parser.add_argument("--target_channel", type=list,
                        default=TARGET_CHANNEL, help="Selected channels.")
    parser.add_argument("--train_ratio", type=float,
                        default=TRAIN_RATIO, help="Train ratio")
    parser.add_argument("--valid_ratio", type=float,
                        default=VALID_RATIO, help="Validate ratio.")
    parser.add_argument("--norm_each_channel", type=float, help="Validate ratio.")
    args = parser.parse_args()

    # print args
    print("-"*(20+45+5))
    for key, value in sorted(vars(args).items()):
        print("|{0:>20} = {1:<45}|".format(key, str(value)))
    print("-"*(20+45+5))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.norm_each_channel = True
    generate_data(args)
    args.norm_each_channel = False
    generate_data(args)
