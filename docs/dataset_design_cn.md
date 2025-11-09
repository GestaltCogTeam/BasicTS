# 📦 数据集设计

## 🎸 新特性

从1.0版本开始，BasicTS采用**数据解耦**的设计方案，用户**可以使用任意数据结构的数据集**，只需继承`BasicTSDataset`基类实现自定义的数据集读取逻辑即可。

从1.0版本开始，BasicTS不再将数据和时间戳整合在一个四维张量中存储（\[batch_size, seq_len, num_features, num_timestamps + 1\]），改为使用两个三维张量，**显著降低了显存占用**。
- 时序数据： \[batch_size, seq_len, num_features\]；
- 时间戳数据：\[batch_size, num_features, num_timestamps\]。

## 📊 内置数据集

## ⏬ 数据下载

要开始使用内置数据集，请先从 [Google Drive](https://drive.google.com/file/d/1m8jh1z4VNMgQ49DRwywyvYYgs3G5WBsB/view?usp=sharing) or [百度网盘](https://pan.baidu.com/s/1UcZCCKPCeS7mHSnCO4-COA?pwd=j9ev) 下载 `datasets.zip` 文件。下载后，将文件解压至 `datasets/` 目录：

```bash
cd /path/to/project
unzip /path/to/all_data.zip -d datasets/
```

这是BasicTS默认的数据集保存路径，当然，您也可以将数据集放在任意其他路径下，并在`dataset_params`中的`data_file_path`字段显式的提供根路径。

这些数据集已经过预处理，可以直接使用。

未来将会支持在线下载内置数据集，该功能目前正在开发中。

## 🔬 使用内置数据集

内置数据集通常配合BasicTS内置支持的数据集类使用，内置数据类也是配置中的默认选项。
内置数据集类：
- **预测任务**：`BasicTSForecastingDataset`；
- **分类任务**：`UEADataset`；
- **插补任务**：`BasicTSImputationDataset`。

这些内置数据集类包括以下参数：
- `dataset_name` (str): 数据集的名称。
- `input_len` (int): 输入序列的长度，即历史数据点的数量。
- `output_len` (int): （仅预测任务）输出序列的长度，即需要预测的未来数据点的数量。
- `mode` (BasicTSMode | str): 数据集的模式，"TRAIN", "VAL"或"TEST"，指示其用于训练、验证还是测试。由runner统一指定，无需手动赋值。
- `use_timestamps (bool)`: 是否使用时间戳的标志，默认为False。
- `local (bool)`: 数据集是否在本地。（开发中）
- `data_file_path` (str | None): 包含时间序列数据文件的路径。默认为 "datasets/{name}"。
- `memmap` (bool): 是否使用内存映射加载数据集的标志。开启时节省内存但会降低训练速度，因此建议仅在数据集极大时使用。默认为False。

通常来说，默认设置下使用内置数据集，只需在配置类中指定`dataset_name`、`input_len`以及`output_len`（预测任务）即可。

## 💿 数据格式

**在BasicTS中，数据集提供的数据需要遵循标准的格式**。`__get_item__`方法应返回一个包含以下项目的字典：
- `inputs`：输入数据，形状为\[batch_size, input_len, num_features\]的`torch.Tensor`；
- `targets`：目标数据（可选）。一个`torch.Tensor`。对于预测和插补任务，形状为\[batch_size, output_len, num_features\]，对于分类任务，形状为\[batch_size, num_classes\]，对于自监督任务，不需要该键；
- `inputs_timestamps`：输入数据的时间戳（可选），形状为\[batch_size, input_len, num_timestamps\]的`torch.Tensor`；
- `targets_timestamps`：输入数据的时间戳（可选），形状为\[batch_size, output_len, num_timestamps\]的`torch.Tensor`。

## 🧑‍🍳 如何添加或自定义数据集

您可以通过以下三步使用您自定义的数据集：
1. 编写数据集类，继承`BasicTSDataset`基类，基类包含三个字段：`dataset_name`，`mode`，`memmap`。
2. 自定义实现您的数据读取和预处理逻辑，实现`__get_item__`和`__len__`方法。请注意，虽然数据实际的存储结构可以是任意的，但`__get_item__`方法返回的数据项应该遵循上文提到的规范。
3. 如果需要使用缩放器对数据做归一化，还需重写data方法（property）。该方法用于向缩放器提供一个待归一化数据的视图（np.ndarray），使缩放器学习整个训练集的分布。
4. 在配置类中修改`dataset_type`字段为您自己的数据集类，并设置相应的`dataset_params`。

## 🧑‍💻 进一步探索

- **🎉 [快速上手](./getting_started_cn.md)**
- **💡 [了解 BasicTS 的设计理念](./overall_design_cn.md)**
- **📦 [探索数据集设计并自定义数据集](./dataset_design_cn.md)**
- **🛠️ [了解数据缩放器设计并创建自定义缩放器](./scaler_design_cn.md)**
- **🧠 [深入了解模型设计并构建自定义模型](./model_design_cn.md)**
- **📉 [了解评估指标设计并开发自定义损失函数与评估指标](./metrics_design_cn.md)**
- **🏃‍♂️ [掌握执行器设计并创建自定义执行器](runner_and_pipeline_cn.md)**
- **📜 [解析配置文件设计并自定义配置](./config_design_cn.md)**
- **🎯 [探索使用BasicTS进行时间序列分类](./time_series_classification_cn.md)**
- **🔍 [探索多种基线模型](../baselines/)**
