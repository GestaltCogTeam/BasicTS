# 📉 Time Series Classification Task

## ✨ Introduction
In the workflow of time series classification, the following key components are usually involved:

- **Dataset**: Defines how to read the dataset and generate samples. (located at `basicts.data`)
- **Metrics**: Defines the evaluation metrics and loss functions. (located at `basicts.metrics.cls_metrics`)
- **Runner**: As the core module of BasicTS, it coordinates the entire training process. The runner integrates the dataset, model architecture, and evaluation metrics, and provides multiple functionalities such as multi-GPU training, distributed training, logging, automatic model saving, gradient clipping, etc. (located at `basicts.runner`)
- **Model**: Defines the model architecture and its forward propagation process.

Below, we will introduce in detail how to use BasicTS for time series classification.  

---

## 🎯 Model Preparation
In BasicTS’s time series classification task, the model input is **inputs** and the output is **prediction**.

- **inputs**: the input sequence, with shape `[batch_size, seq_len, num_nodes, num_features]`, where the last dimension stores additional timestamp information.
- **prediction**: the predicted logits, which contain the unnormalized probabilities for each class, with shape `[batch_size, num_classes]`.

Therefore, in the classification model’s `forward` function, you should implement the mapping from the input dimensions to the output prediction dimensions.

---

## 📦 Dataset

### UEA Dataset
BasicTS provides support for the **UEA dataset**. You can directly use the preprocessing scripts and dataset classes provided in BasicTS to load the UEA dataset.  
The UEA dataset is a commonly used benchmark for time series classification, containing 30 subsets, each divided into training and testing sets.

Steps to use the UEA dataset:
1. Download the UEA dataset from our [Google Drive](https://drive.google.com/file/d/1JGXxKlm6N5JFT7pXn3bb9ntghB8joSV7/view?usp=sharing), [Baidu Cloud](https://pan.baidu.com/s/1FFp0TS-oJyxvqAgyE8_vhA?pwd=six1), or the [official website](https://www.timeseriesclassification.com/).
2. Place the dataset under the path `datasets/raw_data`, e.g., a subset path will look like `datasets/raw_data/UEA/xxx.ts`.
3. Use the provided preprocessing script to convert the UEA dataset into the BasicTS dataset format:
   ```bash
   python scripts/data_preparation/UEA/generate_training_data.py
   ```
   The default preprocessing workflow in BasicTS is as follows: 1) Handle missing values using linear interpolation; 2) Pad sequences to the same length using NaN values; 3) Apply z-score normalization independently for each variable (since different variables may have very different value distributions); 4) Replace NaN values with zeros to avoid affecting the forward process.

    You can also modify the specific preprocessing steps in generate_training_data.py.

Since the UEA dataset does not provide a validation set, we follow the common community practice of using the official test set as the validation set for model selection.

### Other Datasets
If you want to use your own dataset, you can:

1. Write a preprocessing script that handles missing values, normalization, timestamp conversion, etc., and save the processed files in the following format. You can then directly use `TimeSeriesClassificationDataset` and `SimpleTimeSeriesClassificationRunner`.

```
datasets
      ├─Your dataset
      |    ├─train_inputs.npy   // shape [num_samples, seq_len, num_nodes, num_features]
      |    ├─train_labels.npy   // shape [num_samples,]
      |    ├─valid_inputs.npy   // shape [num_samples, seq_len, num_nodes, num_features]
      |    ├─valid_labels.npy   // shape [num_samples,]
      |    ├─test_inputs.npy    // shape [num_samples, seq_len, num_nodes, num_features]
      |    ├─test_labels.npy    // shape [num_samples,]
      |    ├─desc.json          // metadata in JSON format
      ├─(Other datasets...)
```

2. Write a custom Dataset class for your own data format.

# 📝 Configuration File
Compared with the configuration file for forecasting tasks, the main differences are:

* **Dataset Configuration:** Load necessary information from desc.json and configure as follows:
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

* **Metric and loss configuration:**

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

  CFG.TRAIN.LOSS = nn.CrossEntropyLoss() # can be omitted, default is cross entropy loss
  ```

* **Scaler Configuration**: Since classification tasks usually do not require de-normalization, normalization is applied during preprocessing (for UEA datasets), and no Scaler is configured here.

* **Runner Configuration:**

   ```python
   from basicts.runners import SimpleTimeSeriesClassificationRunner
   ...
   CFG.RUNNER = SimpleTimeSeriesClassificationRunner
   ```

# 🚀 Run!
The training process is exactly the same as in forecasting tasks. You just need to run the following command:

```bash
python experiments/train.py -c 'your/config' -g (your gpu)
```

## 🧑‍💻 Explore Further

- **🎉 [Getting Stared](./getting_started.md)**
- **💡 [Understanding the Overall Design Convention of BasicTS](./overall_design.md)**
- **📦 [Exploring the Dataset Convention and Customizing Your Own Dataset](./dataset_design.md)**
- **🛠️ [Navigating The Scaler Convention and Designing Your Own Scaler](./scaler_design.md)**
- **🧠 [Diving into the Model Convention and Creating Your Own Model](./model_design.md)**
- **📉 [Examining the Metrics Convention and Developing Your Own Loss & Metrics](./metrics_design.md)**
- **🏃‍♂️ [Mastering The Runner Convention and Building Your Own Runner](./runner_design.md)**
- **📜 [Interpreting the Config File Convention and Customizing Your Configuration](./config_design.md)**
- **🎯 [Exploring Time Series Classification with BasicTS](./time_series_classification_cn.md)**
- **🔍 [Exploring a Variety of Baseline Models](../baselines/)**
