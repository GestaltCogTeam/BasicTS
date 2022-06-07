# 数据预处理 （以PEMS04数据集为例）

[TOC]

本文档以PEMS04为例，介绍BasicTS的数据预处理过程。包括：原始数据格式、预处理过程、预处理后数据格式。

PEMS04数据集的预处理代码位于`scripts/data_preparation/PEMS04/generate_training_data.py`。

您可以通过借鉴PEMS04数据集的预处理，添加您自己的数据集。

## 1 原始数据

PEMS04数据集来自于交通系统中，共包含来自307个交通传感器的数据。

原始数据位于`datasets/raw_data/PEMS04/PEMS04.npz`，它是一个`[16992, 307, 3]`大小的numpy数组。

其中，16992代表时间序列有16992个时间片，307代表总共有来自307个传感器的307条时间序列，3代表传感器每次采样三种特征。

## 2 预处理过程

时间序列的训练样本通常由一个长度为P+F的滑窗在原始时间序列上滑动得到。
其中，前P个时刻作为历史数据，后F个时刻作为未来数据。

### 2.1 预处理参数

- `output_dir`: 预处理后文件存储位置。
- `data_file_path`： 原生数据位置。
- `graph_file_path`： 图结构数据位置（图结构是非必须的，假如您的数据集没有自带图结构或者您不知道如何构造图结构，可以忽略这一个参数）。
- `history_seq_len`： 历史数据长度，即P的大小。
- `future_seq_len`： 未来数据长度，即F的大小。
- `steps_per_day`： 每天时间片的数量，和采样频率有关。例如每5分钟采样一次，那么该值为288。
- `dow`： 是否添加day in week特征。
- `C`： 选择要使用的特征维度。例如在PEMS04中，我们只需要使用传感器采集的3中特征数据值的第一个维度，所以`C=[0]`。
- `train_ratio`：训练集占总样本量的比例。
- `valid_ratio`：验证集占总样本量的比例。

### 2.2 主要的预处理过程

1. 读取原始数据

```python
import numpy as np
data = np.load(args.data_file_path)['data']     # 大小: [16992, 307, 3]
```

2. 根据原始时间序列的长度和`history_seq_len`和`future_seq_len`的大小，计算总样本的数量，并进一步计算训练、验证、测试样本的数量

```python
num_samples = L - (history_seq_len + future_seq_len) + 1    # 总样本数量
train_num_short = round(num_samples * train_ratio)          # 训练样本数量
valid_num_short = round(num_samples * valid_ratio)          # 验证样本数量
test_num_short  = num_samples - train_num_short - valid_num_short   # 测试样本数量
```

3. 产生样本的index list

对于给定的时刻`t`，它的index是：[t-history_seq_len, t, t+future_seq_len]

```python
index_list      = []
for t in range(history_seq_len, num_samples + history_seq_len):
    index = (t-history_seq_len, t, t+future_seq_len)
    index_list.append(index)
```

4. 数据归一化

不同的数据有不同的量级PEMS04数据集的量级在0到数百之间。

因此，对数据集归一化进行是必须的。

数据归一化最常用的是Z-score归一化，当然也有min-max归一化等其他方法。

PEMS04数据集使用Z-Score归一化函数`standard_transform`。

```python
scaler = standard_transform                         # 归一化函数工具
data_norm = scaler(data, output_dir, train_index)   
# data_norm：归一化后的数据
# output_dir: 用来保存归一化过程中产生的一些参数方便以后使用，例如均值和方差大小。
# train_index： 我们只在训练样本上计算归一化的参数。
```

5. 添加额外的数据

通常我们会为数据集添加一些额外特征。例如PEMS04数据集中，我们为它添加了两个时间特征：tid和dow。

tid是一个大小在[0, 1]范围内的特征，它代表了每天的当前的时间 (time in day)。例如每天有288个时间片，那么每天的第10个时间片的tid特征的值就是`10/288`。

dow代表当前时间是周几（day of week），因此他的选值自然就是{0, 1, 2, 3, 4, 5, 6}中的一个。

需要注意的是，PEMS04数据集可能没有包含绝对时刻，因此我们只能计算相对的时刻：即假设第一个时间片的特征是`tid=0, dow=0`，然后顺序向后计算所有时间片的特征。

```python
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
raw_data = np.concatenate(feature_list, axis=-1)        # 添加完external特征后的数据
```

6. 保存预处理后的数据

```python
# 保存index
index = {}
index['train'] = train_index
index['valid'] = valid_index
index['test']  = test_index
pickle.dump(index, open(output_dir + "/index.pkl", "wb"))

# 保存预处理后的data
data = {}
data['raw_data'] = raw_data
pickle.dump(data, open(output_dir + "/data.pkl", "wb"))

# 保存图结构
# 假如没有的话，可以跳过
if os.path.exists(args.graph_file_path):
    shutil.copyfile(args.graph_file_path, output_dir + '/adj_mx.pkl')      # copy models
else:
    generate_adj_PEMS04()
    shutil.copyfile(args.graph_file_path, output_dir + '/adj_mx.pkl')      # copy models
```

## 3 预处理后的数据

数据的存储形式的规定可以参考[data_preparation_CN.md](docs/DataFormat_CN.md)。

预处理后的数据会被保存在`datasets/PEMS04/`中。

一下所有文件都可以使用`utils/serialization.py`中的`load_pkl`函数读取。

### 3.1 data.pkl

字典类型。`data['processed_data']`保存着预处理后的数据（数组）。

### 3.2 index.pkl

字典类型。产生的训练、验证、测试的index list。

```python
index['train']          # train dataset的index list
index['valid']          # valid dataset的index list
index['test']           # test  dataset的index list
```

### 3.3 scaler.pkl

字典类型，保存着归一化函数以及需要使用的一些参数。例如：

```python
scaler['func']      # 归一化函数
scaler['args']      # 归一化函数需要使用到的参数，字典类型。例如 {"mean": mean, "std": std}.
```

### 3.4 adj_mx.pkl

预定义的邻接矩阵。假如您的数据集没有自带图结构或者您不知道如何为他构造图结构，可以忽略这一个参数。
