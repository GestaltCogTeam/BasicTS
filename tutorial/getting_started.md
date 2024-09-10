# 🎉 Getting Started

Welcome to the BasicTS tutorial! This guide will walk you through the steps of training and evaluating a model using BasicTS.

Before diving in, let’s take a moment to introduce BasicTS.

***What is BasicTS?***

> [!IMPORTANT]  
> BasicTS is a powerful and flexible tool designed specifically for time series forecasting. Whether you are new to this field or an experienced professional, BasicTS provides reliable support. With BasicTS, you can effortlessly build, train, and evaluate your time series forecasting models. You can also compare the performance of various models to find the best solution. We have integrated over 30 algorithms and 20 datasets, with more being added continuously.

***Who Should Use BasicTS?***

> [!IMPORTANT]  
> BasicTS is perfect for both beginners and experts. For beginners looking to enter the world of time series forecasting, BasicTS allows you to quickly grasp the basic pipeline and build your own forecasting model. For experts, BasicTS offers a robust platform for rigorous model comparison, ensuring precise research and development.

***Core Features***

> [!IMPORTANT]  
> Two key features define BasicTS: **fairness** and **scalability**. All models are trained and evaluated under the same conditions, eliminating biases introduced by external factors. This ensures trustworthy comparisons. Additionally, BasicTS is highly scalable, allowing customization of datasets, model structures, and metrics according to your needs. For example, to add a learning rate scheduler, simply specify `CFG.TRAIN.LR_SCHEDULER.TYPE = 'MultiStepLR'` in the configuration file.

Now, let’s begin our journey and explore how to implement your time series forecasting projects with BasicTS!

## ⏬ Cloning the Repository

First, clone the BasicTS repository:

```bash
cd /path/to/your/project
git clone https://github.com/zezhishao/BasicTS.git
```

## 💿 Installing Dependencies

### Operating System

We recommend using BasicTS on Linux systems (e.g., Ubuntu or CentOS).

### Python

Python 3.6 or higher is required (3.8 or higher is recommended).

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) to create a virtual Python environment.

### PyTorch

BasicTS is flexible regarding the PyTorch version. You can [install PyTorch](https://pytorch.org/get-started/previous-versions/) according to your Python version. We recommend using `pip` for installation.

### Other Dependencies

After ensuring PyTorch is installed correctly, you can install the other dependencies:

```bash
pip install -r requirements.txt
```

### Example Setups

#### Example 1: Python 3.9 + PyTorch 1.10.0 + CUDA 11.1

```bash
# Install Python
conda create -n BasicTS python=3.9
conda activate BasicTS
# Install PyTorch
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
# Install other dependencies
pip install -r requirements.txt
```

#### Example 2: Python 3.11 + PyTorch 2.3.1 + CUDA 12.1

```bash
# Install Python
conda create -n BasicTS python=3.11
conda activate BasicTS
# Install PyTorch
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
# Install other dependencies
pip install -r requirements.txt
```

## 📦 Downloading Datasets

You can download the `all_data.zip` file from [Google Drive](https://drive.google.com/drive/folders/14EJVODCU48fGK0FkyeVom_9lETh80Yjp?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1shA2scuMdZHlx6pj35Dl7A?pwd=s2xe). Unzip the files to the `datasets/` directory:

```bash
cd /path/to/BasicTS # not BasicTS/basicts
unzip /path/to/all_data.zip -d datasets/
```

These datasets have been preprocessed and are ready for use.

> [!NOTE]  
> The `data.dat` file is an array in `numpy.memmap` format that stores the raw time series data with a shape of [L, N, C], where L is the number of time steps, N is the number of time series, and C is the number of features.
> 
> The `desc.json` file is a dictionary that stores the dataset’s metadata, including the dataset name, domain, frequency, feature descriptions, regular settings, and null values.
> 
> Other files are optional and may contain additional information, such as `adj_mx.pkl`, which represents a predefined prior graph between the time series.

> [!NOTE]  
> If you are interested in the preprocessing steps, you can refer to the [preprocessing script](../scripts/data_preparation) and `raw_data.zip`.

## 🎯 Quick Tutorial: Train & Evaluate Your Model in Three Steps

### Step 1: Define Your Model

The `forward` function should follow the conventions of BasicTS. An example of the Multi-Layer Perceptron (`MLP`) model can be found in [examples/arch.py](../examples/arch.py).

### Step 2: Define Your Runner

BasicTS provides a unified and standard pipeline in `basicts.runner.BaseTimeSeriesForecastingRunner`. You still need to define the specific forward process in the `forward` function within the **runner**. Fortunately, BasicTS provides a ready-to-use implementation in `basicts.runner.SimpleTimeSeriesForecastingRunner`, which can handle most situations. The runner for the `MLP` model can use this built-in runner.

### Step 3: Configure Your Configuration File

All pipeline details and hyperparameters can be configured in a `.py` file. This configuration file allows you to import your model and runner and set all the options such as model, runner, dataset, scaler, optimizer, loss, and other hyperparameters. An example configuration file for the `MLP` model on the `PEMS08` dataset can be found in [examples/regular_config.py](../examples/regular_config.py).

> [!NOTE]  
> The configuration file is the core of training and evaluation in BasicTS. [`Examples/complete_config.py`](../examples/complete_config.py) outlines all the options available for configuration.

## 🥳 Run It!

`basicts.launch_training` is the entry point for training. You can run the following command to train your model:

- **Train the MLP Model Mentioned Above**

    ```bash
    python experiments/train.py -c examples/regular_config.py -g 0
    ```

or:

- **Reproducing Other Built-in Models**

    BasicTS provides a variety of built-in models. You can reproduce these models with the following command:

    ```bash
    python experiments/train.py -c baselines/${MODEL_NAME}/${DATASET_NAME}.py --gpus '0'
    ```

    Replace `${DATASET_NAME}` and `${MODEL_NAME}` with any supported models and datasets. For example, to run Graph WaveNet on the METR-LA dataset:

    ```bash
    python experiments/train.py -c baselines/GWNet/METR-LA.py --gpus '0'
    ```

## How to Evaluate Your Model

`basicts.launch_evaluation` is the entry point for evaluation. You can run the following command to evaluate your model:

```bash
python experiments/evaluate.py -cfg {CONFIG_FILE}.py -ckpt {CHECKPOINT_PATH}.pth -g 0
```

## 🧑‍💻 Explore Further

This tutorial has equipped you with the fundamentals to get started with BasicTS, but there’s much more to discover. Before delving into advanced topics, let’s take a closer look at the structure of BasicTS:

<div align="center">
  <img src="figures/DesignConvention.jpeg" height=350>
</div>

The core components of BasicTS include `Dataset`, `Scaler`, `Model`, `Metrics`, `Runner`, and `Config`. To streamline the debugging process, BasicTS operates as a localized framework, meaning all the code runs directly on your machine. There’s no need to pip install basicts; simply clone the repository, and you’re ready to run the code locally.

Below are some advanced topics and additional features to help you maximize the potential of BasicTS:

- **🎉 [Getting Stared](./getting_started.md)**
- **💡 [Understanding the Overall Design Convention of BasicTS](./overall_design.md)**
- **📦 [Exploring the Dataset Convention and Customizing Your Own Dataset](./dataset_design.md)**
- **🛠️ [Navigating The Scaler Convention and Designing Your Own Scaler](./scaler_design.md)**
- **🧠 [Diving into the Model Convention and Creating Your Own Model](./model_design.md)**
- **📉 [Examining the Metrics Convention and Developing Your Own Loss & Metrics](./metrics_design.md)**
- **🏃‍♂️ [Mastering The Runner Convention and Building Your Own Runner](./runner_design.md)**
- **📜 [Interpreting the Config File Convention and Customizing Your Configuration](./config_design.md)**
- **🔍 [Exploring a Variety of Baseline Models](../baselines/)**
