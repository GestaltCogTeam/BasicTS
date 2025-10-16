# 📉 时间序列分类任务

## ✨简介

在时间序列分类流程中，通常包含以下几个关键部分：

- **数据集 (Dataset)**：定义读取数据集和生成样本的方式。（位于 `basicts.data`）
- **评估指标 (Metrics)**：定义模型评估的指标和损失函数。（位于 `basicts.metrics.cls_metrics`）
- **执行器 (Runner)**：作为 BasicTS 的核心模块，负责协调整个训练过程。执行器集成了数据集、模型架构和评估指标等组件，提供了多种功能支持，如多 GPU 训练、分布式训练、日志记录、模型自动保存、梯度裁剪等。（位于 `basicts.runner`）
- **模型结构 (Model)**：定义模型架构及其前向传播过程。

下面，我们将详细介绍如何使用BasicTS进行时间序列分类。

## 🎯模型准备

在BasicTS的时间序列分类任务中，模型的输入为`inputs`，返回预测结果`prediction`。

- `inputs`为输入序列，维度为[batch_size, seq_len, num_nodes, num_features]，其中最后一维保存额外的时间戳信息）；
- `prediction`为预测的logits，包含每个类别未归一化的概率值，维度为[batch_size, num_classes]。

因此，在分类模型的`forward`函数中，应实现将指定维度的`inputs`转化为指定维度的`prediction`。

## 📦数据集

### UEA数据集

BasicTS 提供了对 **UEA数据集**的支持，您可以直接使用 BasicTS 中的预处理脚本和数据集类来加载 UEA 数据集。UEA 数据集是一个常用的时间序列分类数据集，包含30个子集，每个数据集被分为训练集和测试集。

使用UEA数据集的流程：

1. 在我们的[Google云](https://drive.google.com/file/d/1JGXxKlm6N5JFT7pXn3bb9ntghB8joSV7/view?usp=sharing)，[百度云](https://pan.baidu.com/s/1FFp0TS-oJyxvqAgyE8_vhA?pwd=six1)，或[官方网站](https://www.timeseriesclassification.com/)下载UEA数据集。

2. 将UEA数据集放至`datasets/raw_data`路径下，此时一个子集的路径为`datasets/raw_data/UEA/xxx.ts`。

3. 使用BasicTS提供的预处理脚本，将UEA数据集转换为BasicTS的数据集格式。

   ```bash
   python scripts/data_preparation/UEA/generate_training_data.py
   ```

   BasicTS默认的数据预处理流程为：1）使用线性插值处理缺失值；2）使用NaN值将序列填充到相同长度；3）使用z-score归一化对每个变量分别进行归一化（由于不同变量间数值分布可能差异很大）；4）将NaN值改为0值，从而不影响forward。

   您也可以在`generate_training_data.py`中修改具体的预处理过程。

由于UEA数据集没有提供验证集，我们参考社区主流的实现方式，将官方提供的测试集作为验证集选择模型。

### 其他数据集

若您想要使用自己的数据集，可以：

1. 写一个预处理程序，包含数据的缺失值处理、归一化、时间戳转换等操作，将文件按指定格式处理至如下路径，并直接使用`TimeSeriesClassificationDataset`和`SimpleTimeSeriesClassificationRunner`。

   ```
   datasets
      ├─Your dataset
      |    ├─train_inputs.npy // 维度为[num_samples, seq_len, num_nodes, num_features]
      |    ├─train_labels.npy // 维度为[num_samples,]
      |    ├─valid_inputs.npy // 维度为[num_samples, seq_len, num_nodes, num_features]
      |    ├─valid_labels.npy // 维度为[num_samples,]
      |    ├─test_inputs.npy // 维度为[num_samples, seq_len, num_nodes, num_features]
      |    ├─test_labels.npy // 维度为[num_samples,]
      |	├─desc.json // json格式的元数据
      ├─(Other datasets...)
   ```

2. 使用您自己的数据格式，写一个Dataset类。

## 📝配置文件

相较于时序预测任务的配置文件，主要区别如下：

* **数据集的配置**：从`desc.json`中读取必要信息，并进行如下配置。

  ```python
  from basicts.data import UEADataset
  from basicts.utils import load_dataset_desc
  
  DATA_NAME = 'JapaneseVowels'  # Dataset name
  desc = load_dataset_desc(os.path.join('UEA', DATA_NAME))
  INPUT_LEN = desc['seq_len']
  NUM_CLASSES = desc['num_classes']
  NUM_NODES = desc['num_nodes']
  ...
  # Dataset settings
  CFG.DATASET.NAME = DATA_NAME
  CFG.DATASET.TYPE = UEADataset
  CFG.DATASET.NUM_CLASSES = NUM_CLASSES
  CFG.DATASET.PARAM = EasyDict({
      'dataset_name': DATA_NAME,
      'train_val_test_ratio': None,
      # 'mode' is automatically set by the runner
  })
  ```

* **指标和损失函数的配置**：

  ```python
  from basicts.metrics import accuracy
  from torch import nn
  
  NULL_VAL = 0.0
  ...
  CFG.METRICS.FUNCS = EasyDict({
                                  'Accuracy': accuracy,
                               })
  CFG.METRICS.TARGET = 'Accuracy'
  CFG.METRICS.NULL_VAL = NULL_VAL

  CFG.TRAIN.LOSS = nn.CrossEntropyLoss() # 可以省略，默认使用交叉熵损失函数
  ```

* **缩放器（Scaler）的配置**：由于分类任务通常不需要反归一化，因此我们在预处理中对UEA数据集进行归一化，不再配置Scaler。

* **执行器（Runner）的配置**：

  ```python
  from basicts.runners import SimpleTimeSeriesClassificationRunner
  ...
  CFG.RUNNER = SimpleTimeSeriesClassificationRunner
  ```

## 🚀运行！

和预测任务完全相同，您只需要运行下列命令：

```bash
python experiments/train.py -c 'your/config' -g (your gpu)
```

## 🧑‍💻 进一步探索

- **🎉 [快速上手](./getting_started_cn.md)**
- **💡 [了解 BasicTS 的设计理念](./overall_design_cn.md)**
- **📦 [探索数据集设计并自定义数据集](./dataset_design_cn.md)**
- **🛠️ [了解数据缩放器设计并创建自定义缩放器](./scaler_design_cn.md)**
- **🧠 [深入了解模型设计并构建自定义模型](./model_design_cn.md)**
- **📉 [了解评估指标设计并开发自定义损失函数与评估指标](./metrics_design_cn.md)**
- **🏃‍♂️ [掌握执行器设计并创建自定义执行器](./runner_design_cn.md)**
- **📜 [解析配置文件设计并自定义配置](./config_design_cn.md)**
- **🎯 [探索使用BasicTS进行时间序列分类](./time_series_classification_cn.md)**
- **🔍 [探索多种基线模型](../baselines/)**