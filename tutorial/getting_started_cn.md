# 🎉 快速上手

欢迎使用 BasicTS 教程！本指南将带您逐步完成使用 BasicTS 训练和评估模型的过程。

在深入之前，我们先简单介绍一下 BasicTS。

***什么是 BasicTS？***

> [!IMPORTANT]  
> BasicTS 是一个专为时间序列预测设计的强大且灵活的工具。无论您是该领域的新手，还是经验丰富的专业人士，BasicTS 都能为您提供可靠的支持。使用 BasicTS，您可以轻松构建、训练和评估时间序列预测模型，还能比较各种模型的性能，找到最佳解决方案。我们已经集成了超过30种算法和20个数据集，并在持续添加更多内容。

***谁应该使用 BasicTS？***

> [!IMPORTANT]  
> BasicTS 非常适合初学者和专家使用。对于想要进入时间序列预测领域的初学者来说，BasicTS 能帮助你快速掌握基本流程并构建自己的预测模型。对于专家来说，BasicTS 提供了一个强大的平台，用于进行严格的模型比较，确保精准的研究与开发。

***核心功能***

> [!IMPORTANT]  
> BasicTS 有两个关键特性：**公平性** 和 **可扩展性**。所有模型都在相同条件下训练和评估，消除了由外部因素引入的偏差，确保了可靠的比较。此外，BasicTS 具有高度的可扩展性，允许根据需要自定义数据集、模型结构和评估指标。例如，您只需在配置文件中指定 `CFG.TRAIN.LR_SCHEDULER.TYPE = 'MultiStepLR'`，即可添加学习率调度器。

现在，让我们开始探索如何通过 BasicTS 实现您的时间序列预测项目吧！

## ⏬ 克隆仓库

首先，克隆 BasicTS 仓库：

```bash
cd /path/to/your/project
git clone https://github.com/zezhishao/BasicTS.git
```

## 💿 安装依赖项

### 操作系统

我们建议在 Linux 系统（如 Ubuntu 或 CentOS）上使用 BasicTS。

### Python

需要 Python 3.6 或更高版本（建议使用 3.8 或更高版本）。

我们推荐使用 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 [Anaconda](https://www.anaconda.com/) 来创建虚拟 Python 环境。

### PyTorch

BasicTS 对 PyTorch 版本非常灵活。您可以根据 Python 版本[安装 PyTorch](https://pytorch.org/get-started/previous-versions/)。我们建议使用 `pip` 进行安装。

### 其他依赖项

确保 PyTorch 正确安装后，您可以安装其他依赖项：

```bash
pip install -r requirements.txt
```

### 示例设置

#### 示例 1：Python 3.11 + PyTorch 2.3.1 + CUDA 12.1 (推荐)

```bash
# 安装 Python
conda create -n BasicTS python=3.11
conda activate BasicTS
# 安装 PyTorch
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
# 安装其他依赖项
pip install -r requirements.txt
```

#### 示例 2：Python 3.9 + PyTorch 1.10.0 + CUDA 11.1

```bash
# 安装 Python
conda create -n BasicTS python=3.9
conda activate BasicTS
# 安装 PyTorch
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
# 安装其他依赖项
pip install -r requirements.txt
```

## 📦 下载数据集

您可以从 [Google Drive](https://drive.google.com/drive/folders/14EJVODCU48fGK0FkyeVom_9lETh80Yjp?usp=sharing) 或 [百度网盘](https://pan.baidu.com/s/1shA2scuMdZHlx6pj35Dl7A?pwd=s2xe) 下载 `all_data.zip` 文件。将文件解压到 `datasets/` 目录：

```bash
cd /path/to/BasicTS # not BasicTS/basicts
unzip /path/to/all_data.zip -d datasets/
```

这些数据集已预处理完毕，可以直接使用。

> [!NOTE]  
> `data.dat` 文件是以 `numpy.memmap` 格式存储的数组，包含原始时间序列数据，形状为 [L, N, C]，其中 L 是时间步数，N 是时间序列数，C 是特征数。
> 
> `desc.json` 文件是一个字典，存储了数据集的元数据，包括数据集名称、领域、频率、特征描述、常规设置和缺失值。
> 
> 其他文件是可选的，可能包含附加信息，如表示时间序列间预定义图结构的 `adj_mx.pkl`。

> [!NOTE]  
> 如果您对预处理步骤感兴趣，可以参考[预处理脚本](../scripts/data_preparation) 和 `raw_data.zip`。

## 🎯 快速教程：三步训练并评估您的模型

### 第一步：定义您的模型

`forward` 函数应该遵循 BasicTS 的规范。多层感知机（`MLP`）模型的示例可以在 [examples/arch.py](../examples/arch.py) 中找到。

### 第二步：定义您的执行器

BasicTS 提供了一个统一的标准化流程，位于 `basicts.runner.BaseTimeSeriesForecastingRunner`。您仍然需要在 **执行器** 中的 `forward` 函数中定义具体的前向过程。

幸运的是，BasicTS 已提供了一个可直接使用的实现，`basicts.runner.SimpleTimeSeriesForecastingRunner`，可处理大多数情况。`MLP` 模型的执行器可以使用这个内置执行器。

### 第三步：配置您的配置文件

所有流程细节和超参数都可以在 `.py` 文件中配置。该配置文件允许您导入模型和执行器，并设置所有选项，如模型、执行器、数据集、数据缩放器、优化器、损失函数和其他超参数。`MLP` 模型在 `PEMS08` 数据集上的配置示例可在 [examples/regular_config.py](../examples/regular_config.py) 中找到。

> [!NOTE]  
> 配置文件是 BasicTS 中训练和评估的核心。[`Examples/complete_config.py`](../examples/complete_config_cn.py) 列出了所有可配置的选项。

## 🥳 运行它！

`basicts.launch_training` 是训练的入口点。您可以运行以下命令来训练您的模型：

- **训练上述提到的 MLP 模型**

    ```bash
    python experiments/train.py -c examples/regular_config.py -g 0
    ```

或者：

- **复现其他内置模型**

    BasicTS 提供了多种内置模型。您可以通过以下命令复现这些模型：

    ```bash
    python experiments/train.py -c baselines/${MODEL_NAME}/${DATASET_NAME}.py --gpus '0'
    ```

    将 `${DATASET_NAME}` 和 `${MODEL_NAME}` 替换为任意支持的数据集和模型。例如，在 METR-LA 数据集上运行 Graph WaveNet 模型：

    ```bash
    python experiments/train.py -c baselines/GWNet/METR-LA.py --gpus '0'
    ```

## 如何评估您的模型

`basicts.launch_evaluation` 是评估的入口点。您可以运行以下命令来评估您的模型：

```bash
python experiments/evaluate.py -cfg {CONFIG_FILE}.py -ckpt {CHECKPOINT_PATH}.pth -g 0
```

## 🧑‍💻 进一步探索

本教程为您提供了 BasicTS 的基础知识，但还有更多内容等待您探索。在深入其他主题之前，我们先更详细地了解 BasicTS 的结构：

<div align="center">
  <img src="figures/DesignConvention.jpeg" height=350>
</div>

BasicTS 的核心组件包括 `Dataset`、`Scaler`、`Model`、`Metrics`、`Runner` 和 `Config`。为简化调试过程，BasicTS 作为一个本地化框架运行，所有代码都直接在您的机器上运行。无需 `pip install basicts`，只需克隆仓库，即可本地运行代码。

以下是一些高级主题和附加功能，帮助您充分利用 BasicTS：


- **🎉 [快速上手](./getting_started_cn.md)**
- **💡 [了解 BasicTS 的设计理念](./overall_design_cn.md)**
- **📦 [探索数据集设计并自定义数据集](./dataset_design_cn.md)**
- **🛠️ [了解数据缩放器设计并创建自定义缩放器](./scaler_design_cn.md)**
- **🧠 [深入了解模型设计并构建自定义模型](./model_design_cn.md)**
- **📉 [了解评估指标设计并开发自定义损失函数与评估指标](./metrics_design_cn.md)**
- **🏃‍♂️ [掌握执行器设计并创建自定义执行器](./runner_design_cn.md)**
- **📜 [解析配置文件设计并自定义配置](./config_design_cn.md)**
- **🔍 [探索多种基线模型](../baselines/)**
