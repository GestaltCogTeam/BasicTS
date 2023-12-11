import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
# 向前退三级
project_dir = os.path.abspath(__file__ + "/../../../..")
data_dir = project_dir + "/datasets/raw_data/M4/"
seasonal_pattern = "Monthly"
output_dir = project_dir + "/datasets/M4_{0}/".format(seasonal_pattern)
processed_data_train = [] # final data
processed_mask_train = []
processed_data_test = []
processed_mask_test = []
train_index = []
test_index = []
scaler = []
future_seq_len = {"Yearly": 6, "Quarterly": 8, "Monthly": 18, "Weekly": 13, "Daily": 14, "Hourly": 48}[seasonal_pattern]
lookback = 2
history_seq_len = future_seq_len * lookback
index_lower_bound = int({"Yearly": 1.5, "Quarterly": 1.5, "Monthly": 1.5, "Weekly": 10, "Daily": 10, "Hourly": 10}[seasonal_pattern] * future_seq_len) # generate index from len(train_data) - index_lower_bound

# read data
train_data = pd.read_csv(data_dir + seasonal_pattern + "-train.csv")
test_data = pd.read_csv(data_dir + seasonal_pattern + "-test.csv")
meta_info = pd.read_csv(data_dir + "M4-info.csv")

def conver_date(start_date):
    # 判断format: '%d-%m-%y %H:%M' or '%Y-%m-%d %H:%M:%S'
    if len(start_date) == 14: format = '%d-%m-%y %H:%M'
    elif len(start_date) == 19: format = '%Y-%m-%d %H:%M:%S'
    else: raise ValueError("start_date should be in format '%d-%m-%y %H:%M' or '%Y-%m-%d %H:%M:%S'")
    return pd.to_datetime(start_date, format=format)

def generate_dataframe(start_time, frequency, the_array):
    # E.g.
    # start_time = '08-01-75 12:00'
    # frequency = 'Yearly'
    if isinstance(start_time, str):
        start_time = conver_date(start_time)
    # date range
    freq = frequency[0] # {'Yearly': 'Y', 'Quarterly': 'Q', 'Monthly': 'M', 'Weekly': 'W', 'Daily': 'D', 'Hourly': 'H'}[frequency]
    date_range = pd.date_range(start=start_time, periods=len(the_array), freq=freq)
    # create the dataframe
    df = pd.DataFrame(index=date_range, data={'Values': the_array})
    return df

def process_one_series(ts_id, ts_train, ts_info):
    """generate data and index for one series.

    Args:
        ts_id (int): time series id in each subset.
        ts_train (pd.Series): time series data.
        info (pd.Series): time series info.
    """
    ts_train = ts_train.tolist()
    mask_train = [1] * len(ts_train)

    # generate padded data
    low = max(1, len(ts_train) - index_lower_bound)
    if low - history_seq_len < 0: left_padding = [0] * (history_seq_len - low)
    else: left_padding = []
    right_padding = [0] * future_seq_len
    ts_train_padded = left_padding + ts_train + right_padding
    # generate mask
    mask_train_padded = [0] * len(left_padding) + mask_train + [0] * len(right_padding)
    # generate temporal features
    start_time = conver_date(ts_info["StartingDate"])
    if seasonal_pattern == "Yearly": start_time_padded = start_time - pd.DateOffset(years=len(left_padding))
    elif seasonal_pattern == "Quarterly": start_time_padded = start_time - pd.DateOffset(months=3*len(left_padding))
    elif seasonal_pattern == "Monthly": start_time_padded = start_time - pd.DateOffset(months=len(left_padding))
    elif seasonal_pattern == "Weekly": start_time_padded = start_time - pd.DateOffset(weeks=len(left_padding))
    elif seasonal_pattern == "Daily": start_time_padded = start_time - pd.DateOffset(days=len(left_padding))
    elif seasonal_pattern == "Hourly": start_time_padded = start_time - pd.DateOffset(hours=len(left_padding))
    else: raise ValueError("seasonal_pattern should be one of ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']")
    df = generate_dataframe(start_time_padded, seasonal_pattern, ts_train_padded)
    df["tod"] = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    df["dow"] = df.index.dayofweek / 7
    df["dom"] = (df.index.day - 1) / 31 # df.index.day starts from 1. We need to minus 1 to make it start from 0.
    df["doy"] = (df.index.dayofyear - 1) / 366 # df.index.month starts from 1. We need to minus 1 to make it start from 0.
    # generate data
    new_data = df.values.tolist()
    # generate index
    index_list = []
    for t in range(low + len(left_padding), len(ts_train_padded) - future_seq_len):
        index_list.append([t - history_seq_len, t, t + future_seq_len])
    # generate scaler
    scaler = None
    return new_data, mask_train_padded, index_list, scaler

# generate training data
for ts_id in tqdm(range(train_data.shape[0])):
    ts_name = train_data.iloc[ts_id, :]["V1"]
    ts_info = meta_info[meta_info["M4id"] == ts_name].iloc[0, :]
    ts_train = train_data.iloc[ts_id, :].drop("V1").dropna()

    ts_data, ts_mask, ts_index, ts_scaler = process_one_series(ts_id, ts_train, ts_info)
    processed_data_train.append(ts_data)
    processed_mask_train.append(ts_mask)
    train_index.append(ts_index)
    scaler.append(ts_scaler)

for ts_id in tqdm(range(test_data.shape[0])):
    ts_name = test_data.iloc[ts_id, :]["V1"]
    ts_info = meta_info[meta_info["M4id"] == ts_name].iloc[0, :]
    ts_test = test_data.iloc[ts_id, :].drop("V1").dropna()
    ts_train_last_sample_index = train_index[ts_id][-1]
    ts_train_last_sample_history = processed_data_train[ts_id][ts_train_last_sample_index[0]+1:ts_train_last_sample_index[1]+1] # last history sample
    ts_train_last_sample_future = processed_data_train[ts_id][ts_train_last_sample_index[1]+1:ts_train_last_sample_index[2]+1] # last future sample, should be all zeros
    ts_train_last_sample_mask = processed_mask_train[ts_id][ts_train_last_sample_index[0]+1:ts_train_last_sample_index[1]+1] # last history sample mask. there might be some mask in the history data when the history_seq_len is large.
    assert sum([_[0] for _ in ts_train_last_sample_future]) == 0
    ts_train_last_sample_future = np.array(ts_train_last_sample_future)
    ts_train_last_sample_future[:, 0] = ts_test.tolist()
    ts_data = ts_train_last_sample_history + ts_train_last_sample_future.tolist()
    processed_data_test.append(ts_data)
    processed_mask_test.append(ts_train_last_sample_mask + [1] * len(ts_test))
    test_index.append([[0, len(ts_train_last_sample_history), len(ts_train_last_sample_history) + len(ts_test)]])
    assert ts_test.shape[0] == future_seq_len, "test data length should be equal to future_seq_len"

# create output dir if not exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# save
import pickle
## save data
with open(output_dir + "/data_in_{0}_out_{1}_rescale_None.pkl".format(history_seq_len, future_seq_len), "wb") as f:
    pickle.dump({"train": processed_data_train, "test": processed_data_test}, f)
## save mask
with open(output_dir + "/mask_in_{0}_out_{1}_rescale_None.pkl".format(history_seq_len, future_seq_len), "wb") as f:
    pickle.dump({"train": processed_mask_train, "test": processed_mask_test}, f)
## save index
with open(output_dir + "/index_in_{0}_out_{1}_rescale_None.pkl".format(history_seq_len, future_seq_len), "wb") as f:
    pickle.dump({"train": train_index, "test": test_index}, f)
## save scaler
# with open(output_dir + "/scaler_in_{0}_out_{1}.pkl".format(history_seq_len, future_seq_len), "wb") as f:
#     pickle.dump({"scaler": scaler}, f)
a = 1
