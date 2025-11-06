# 📦 Dataset Design

## 🎸 New Features

Starting from version 1.0, BasicTS adopts a **data decoupling** design approach, allowing users to **use datasets with any data structure** by simply inheriting from the `BasicTSDataset` base class and implementing custom data loading logic.

From version 1.0 onward, BasicTS no longer stores data and timestamps in a single four-dimensional tensor (\[batch_size, seq_len, num_features, num_timestamps + 1\]). Instead, it uses two separate three-dimensional tensors, **significantly reducing GPU memory usage**:
- Time series data: \[batch_size, seq_len, num_features\]
- Timestamp data: \[batch_size, num_features, num_timestamps\]

## 📊 Built-in Datasets

## ⏬ Data Download

To start using the built-in datasets, first download the `datasets.zip` file from [Google Drive](https://drive.google.com/file/d/1m8jh1z4VNMgQ49DRwywyvYYgs3G5WBsB/view?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1UcZCCKPCeS7mHSnCO4-COA?pwd=j9ev). After downloading, extract the file to the `datasets/` directory:

```bash
cd /path/to/project
unzip /path/to/datasets.zip -d datasets/
```

This is the default dataset storage path for BasicTS. However, you can also place datasets in any other directory and explicitly provide the root path in the `data_file_path` field within `dataset_params`.

These datasets are preprocessed and ready to use.

Online downloading of built-in datasets will be supported in the future; this feature is currently under development.

## 🔬 Using Built-in Datasets

Built-in datasets are typically used with BasicTS's built-in dataset classes, which are also the default options in configurations.

Built-in dataset classes:
- **Forecasting Task**: `BasicTSForecastingDataset`
- **Classification Task**: `UEADataset`
- **Imputation Task**: `BasicTSImputationDataset`

These built-in dataset classes include the following parameters:
- `dataset_name` (str): The name of the dataset.
- `input_len` (int): The length of the input sequence, i.e., the number of historical data points.
- `output_len` (int): (Forecasting task only) The length of the output sequence, i.e., the number of future data points to predict.
- `mode` (BasicTSMode | str): The mode of the dataset, "TRAIN", "VAL", or "TEST", indicating whether it is used for training, validation, or testing. Set by the runner automatically; no manual assignment needed.
- `use_timestamps` (bool): Flag to determine if timestamps should be used. Default is False.
- `local` (bool): Whether the dataset is stored locally. (Under development)
- `data_file_path` (str | None): Path to the file containing the time series data. Defaults to "datasets/{name}".
- `memmap` (bool): Flag to determine if the dataset should be loaded using memory mapping. Enabling this saves memory but slows down training, so it is recommended only for very large datasets. Default is False.

Generally, when using built-in datasets with default settings, you only need to specify `dataset_name`, `input_len`, and `output_len` (for forecasting tasks) in the configuration class.

## 💿 Data Format

**In BasicTS, data provided by datasets must adhere to a standard format.** The `__getitem__` method should return a dictionary containing the following items:
- `inputs`: Input data, a `torch.Tensor` with shape \[batch_size, input_len, num_features\]
- `targets`: Target data (optional). A `torch.Tensor`. For forecasting and imputation tasks, shape is \[batch_size, output_len, num_features\]; for classification tasks, shape is \[batch_size, num_classes\]; for self-supervised tasks, this key is not required
- `inputs_timestamps`: Timestamps for the input data (optional), a `torch.Tensor` with shape \[batch_size, input_len, num_timestamps\]
- `targets_timestamps`: Timestamps for the target data (optional), a `torch.Tensor` with shape \[batch_size, output_len, num_timestamps\]

## 🧑‍🍳 How to Add or Customize a Dataset

You can use your custom dataset by following these three steps:
1. Write a dataset class that inherits from the `BasicTSDataset` base class, which includes three fields: `dataset_name`, `mode`, and `memmap`.
2. Implement your custom data loading and preprocessing logic by implementing the `__getitem__` and `__len__` methods. Note that while the actual storage structure of the data can be arbitrary, the data items returned by the `__getitem__` method should follow the specifications mentioned above.
3. If you need to use a scaler to normalize the data, you must also override the `data` property method. This method provides the scaler with a view of the data to be normalized (as an `np.ndarray`), allowing the scaler to learn the distribution of the entire training set.
4. In the configuration class, modify the `dataset_type` field to your own dataset class and set the corresponding `dataset_params`.

## 🧑‍💻 Explore Further

- **🎉 [Getting Stared](./getting_started.md)**
- **💡 [Understanding the Overall Design Convention of BasicTS](./overall_design.md)**
- **📦 [Exploring the Dataset Convention and Customizing Your Own Dataset](./dataset_design.md)**
- **🛠️ [Navigating The Scaler Convention and Designing Your Own Scaler](./scaler_design.md)**
- **🧠 [Diving into the Model Convention and Creating Your Own Model](./model_design.md)**
- **📉 [Examining the Metrics Convention and Developing Your Own Loss & Metrics](./metrics_design.md)**
- **🏃‍♂️ [Mastering The Runner Convention and Building Your Own Runner](runner_and_pipeline.md)**
- **📜 [Interpreting the Config File Convention and Customizing Your Configuration](./config_design.md)**
- **🎯 [Exploring Time Series Classification with BasicTS](./time_series_classification_cn.md)**
- **🔍 [Exploring a Variety of Baseline Models](../baselines/)**
