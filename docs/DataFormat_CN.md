# 数据格式

- [数据格式](#数据格式)
  - [数据存储形式](#数据存储形式)
  - [为什么要这样存储数据](#为什么要这样存储数据)
  - [Reference](#reference)

## 数据存储形式

为了灵活的读取、预处理数据，我们设计了一种高效、通用的数据读取/预处理流程。

假设原始时间序列为$\mathbf{X}$，时刻$t$处的数据为$\mathbf{X}_t$。
那么在一般情况下，时间序列预测的一个训练样本就是指：在时间$t$时刻，使用历史的$p$个时间片：$\mathbf{X}_{t-p+1}, ..., \mathbf{X}_{t}$个时间片，预测未来$f$个时间片$\mathbf{X}_{t+1}, ..., \mathbf{X}_{t+f}$。
但也有一些特殊情况，时间序列的样本是不连续的，例如许多包含多分辨率思想的工作[1][2].

为了提供**统一、通用**的data pipline，我们采用存储所有样本的index，而非训练样本的方式。
具体来说，预处理代码将为每个数据集产生四个文件：

- `index.pkl`: dict of list
  - keys: train, valid, test
  - values: 每个训练样本的index，分三种情况，用户可以方便地自定义
    - 连续(默认):   [current_time_step_index-p+1, current_time_step_index, current_time_step_index+f]
    - 不连续: [[x, x, ..., x], current_time_step_index, current_time_step_index+f]
    - 其他

- `data.pkl`: dict of list
  - keys: raw_data, other
  - values: 归一化后并且添加好特征后的“原始”时间序列序列或者其他辅助时间序列
    - raw_data: np.array, L x N x C. L: 时间序列总长度, N: 多变量时间序列数量, C: 特征数量
    - other

- `scaler.pkl`: dict of list
  - keys:
    - args: 归一化/反归一化参数，例如mean/std, min/max
    - func: 反归一化函数
  - values

- `adj_mx.pkl`: the pre-defined adjacent matrix

借助numpy强大的功能，这样的操作方式可以在保证速度的情况下，极大地满足可扩展性，满足几乎所有模型的数据读取需求。

## 为什么要这样存储数据

统一的数据读取/预处理流程可以方便地、公平地对比不同的Baseline的效率，更简单的理解代码。

需要注意的是，尽管这种读取方法是高效的，但并不总是最快的。例如，将所有的样本预处理到本地可能更快一些，但那会损失通用能力。

## Reference

[1] Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting\
[2] Learning Dynamics and Heterogeneity of Spatial-Temporal Graph Data for Trafﬁc Forecasting
