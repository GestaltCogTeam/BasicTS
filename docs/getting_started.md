# 🎉 Quick Start

Welcome to the BasicTS tutorial! This guide will walk you through the process of training and evaluating models using BasicTS 1.0 step by step.

Before diving in, let's briefly introduce BasicTS.

***What is BasicTS?***

> [!IMPORTANT]
> BasicTS is a powerful and flexible toolkit designed specifically for time series analysis. Whether you are a novice in the field or an experienced professional, BasicTS provides reliable support for your work. With BasicTS, you can easily build, train, and evaluate time series forecasting models, as well as compare the performance of various models to find the optimal solution. We have integrated over 30 algorithms and 20 datasets, and we are continuously adding more.

***Who should use BasicTS?***

> [!IMPORTANT]
> BasicTS is perfectly suited for both beginners and experts. For beginners looking to enter the field of time series analysis, BasicTS helps you quickly grasp the fundamental workflow and build your own analysis models. For experts, BasicTS provides a robust platform for rigorous model comparison, ensuring precision in research and development.

***Core Features***

> [!IMPORTANT]
> BasicTS has two key characteristics: **Fairness** and **Extensibility**.
> **Fairness**: All models are trained and evaluated under identical conditions, eliminating biases introduced by external factors and ensuring reliable comparisons.
> **Extensibility**: BasicTS is highly extensible, allowing customization of datasets, model architectures, and evaluation metrics as needed. In version 1.0, the extensibility of BasicTS has been significantly enhanced, enabling you to easily customize your own models and datasets according to your requirements.

Now, let's get started on exploring how to realize your time series analysis projects with BasicTS!

## 📦 Install BasicTS

We recommend installing BasicTS on a Linux system (such as Ubuntu or CentOS) with Python 3.8 or higher:

```bash
pip install basicts
```

We recommend using https://docs.conda.io/en/latest/miniconda.html or https://www.anaconda.com/ to create a virtual Python environment.

## 🔧 Install Dependencies

### PyTorch

BasicTS is very flexible regarding PyTorch versions. You can https://pytorch.org/get-started/previous-versions/ according to your Python version. We recommend using `pip` for installation.

### Example Setups

#### Example 1: Python 3.11 + PyTorch 2.5.1 + CUDA 12.4 (Recommended)

```bash
# Install Python
conda create -n BasicTS python=3.11
conda activate BasicTS
# Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

#### Example 2: Python 3.9 + PyTorch 1.10.0 + CUDA 11.1

```bash
# Install Python
conda create -n BasicTS python=3.9
conda activate BasicTS
# Install PyTorch
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## 🔍 Download Datasets

You can download the `datasets.zip` file from [Google Drive](https://drive.google.com/file/d/1m8jh1z4VNMgQ49DRwywyvYYgs3G5WBsB/view?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1UcZCCKPCeS7mHSnCO4-COA?pwd=j9ev). Extract the file to the `datasets/` directory:

```bash
cd /path/to/YourProject # not BasicTS/basicts
unzip /path/to/datasets.zip -d datasets/
```

These datasets are preprocessed and ready to use.

> [!NOTE]
> The `data.dat` file is an array stored in `numpy.memmap` format, containing the raw time series data with shape [L, N, C], where L is the number of time steps, N is the number of time series, and C is the number of features.
>
> The `desc.json` file is a dictionary storing the metadata of the dataset, including the dataset name, domain, frequency, feature descriptions, general settings, and missing values.
>
> Other files are optional and may contain additional information, such as `adj_mx.pkl` representing a predefined graph structure between time series.

> [!NOTE]
> If you are interested in the preprocessing steps, you can refer to the ../scripts/data_preparation and `raw_data.zip`.

## 🎯 Quick Tutorial: Train and Evaluate Your Model in Three Lines of Code

```python
# train.py

from basicts.models.DLinear import DLinear, DLinearConfig
from basicts.configs import BasicTSForecastingConfig
from basicts import BasicTSLauncher

def main():

    # 1. Configure the model
    model_config = DLinearConfig(input_len=336, output_len=336)

    # 2. Configure the task
    cfg = BasicTSForecastingConfig(
        model=DLinear,
        model_config=model_config,
        dataset_name="ETTh1",
        gpus="0",
        ...
    )

    # 3. Launch training
    BasicTSLauncher.launch_training(cfg)

```

### Step 1: Configure Your Model

BasicTS provides a large number of commonly used models in `basicts.models`, which you can use directly. BasicTS uses configuration classes to configure models. Each configuration class contains detailed descriptions of every parameter needed to construct the model. For example, the configuration class for the DLinear model is `DLinearConfig`. You can find the `DLinearConfig` class in ../basicts/models/config/dlinear_config.py.

If you want to use your own model, you need to follow the BasicTS specifications. For details, please see 🧠 ./model_design.md.

### Step 2: Configure Your Task

BasicTS supports various time series tasks, including forecasting, imputation, classification, etc. The task configuration class is the core of BasicTS. Almost all information about a BasicTS task is encapsulated in the task configuration class. Almost all configuration items have commonly used default values. You only need to configure the key parameters (1️⃣ model, 2️⃣ dataset) and modify a few settings (such as batch size, learning rate, etc.) to run the code.

> [!NOTE]
> You can find the configuration classes for each BasicTS task (e.g., the configuration class for the forecasting task is `BasicTSForecastingConfig`) and the meaning and configuration methods of each parameter in ../basicts/configs.

Furthermore, in BasicTS configuration classes, you can also specify callbacks and taskflows to perform additional operations during training (such as curriculum learning) and customize data processing pipelines. For advanced usage of BasicTS configuration classes, please see 📜 ./config_design.md.

### Step 3: Launch Training

`BasicTSLauncher.launch_training` is the entry point for training. Call this method and pass in the task configuration to start training.

> [!NOTE]
> It is important to note that in DDP mode, `BasicTSLauncher.launch_training` needs to be wrapped in `if __name__ == '__main__':` to ensure that each process correctly initializes the model and dataset.

## 🥳 Run It!

In your project directory, run the following command to start training:
```bash
python train.py
```
During training, BasicTS will save the trained models to the `checkpoints/` directory by default and perform evaluation after training is complete (this can be changed via configuration). You can also choose to save evaluation metrics and results to the `checkpoints/` directory.

You can find more runnable examples in the [examples](../examples) directory.
## How to Evaluate Your Model

Of course, you can also manually evaluate the model after training: `BasicTSLauncher.launch_evaluation` is the entry point for evaluation. You can evaluate your model by executing the following Python code.

```python
BasicTSLauncher.launch_evaluation(cfg, "checkpoints/your_checkpoint.pt")
```

## 🧑💻 Further Exploration

This tutorial has provided you with the basics of BasicTS, but there is much more to explore. Before delving into other topics, let's take a closer look at the structure of BasicTS:

<div align="center">
  
</div>

The core components of BasicTS include `Dataset`, `Scaler`, `Model`, `Metrics`, `Runner`, and `Config`.

Here are some advanced topics and additional features to help you make the most of BasicTS:


- **🎉 [Getting Stared](./getting_started.md)**
- **💡 [Understanding the Overall Design Convention of BasicTS](./overall_design.md)**
- **📦 [Exploring the Dataset Convention and Customizing Your Own Dataset](./dataset_design.md)**
- **🛠️ [Navigating The Scaler Convention and Designing Your Own Scaler](./scaler_design.md)**
- **🧠 [Diving into the Model Convention and Creating Your Own Model](./model_design.md)**
- **📉 [Examining the Metrics Convention and Developing Your Own Loss & Metrics](./metrics_design.md)**
- **🏃‍♂️ [Mastering The Runner Convention and Building Your Own Runner](runner_and_pipeline.md)**
- **📜 [Interpreting the Config File Convention and Advanced Configuration](./config_design.md)**
- **🎯 [Exploring Time Series Classification with BasicTS](./time_series_classification_cn.md)**
- **🔍 [Exploring a Variety of Baseline Models](../baselines/)**