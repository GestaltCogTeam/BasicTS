# BasicTS

[![EasyTorch](https://img.shields.io/badge/Developing%20with-EasyTorch-2077ff.svg)](https://github.com/cnstark/easytorch)

## 0. What is BasicTS

BasicTS (**Basic** **T**ime **S**eries) is an open-source PyTorch-based **time series** benchmark and toolbox motivated by [BasicSR](https://github.com/xinntao/BasicSR) [1].
At present, it only focuses on **time series forecasting**, and may add time series classification, anomaly detection, etc., in the future.
BasicTS provides users with a unified, standard pipline (fair, but probably not the fastest), which provide reproduction and fair comparision of popular time series models to inspire new innovations.
BasicTS is developed based on [EasyTorch](https://github.com/cnstark/easytorch) [2], an easy-to-use and powerful open source neural network training framework.

BasicTS (**Basic** **T**ime **S**eries)是一个受[BasicSR](https://github.com/xinntao/BasicSR) [1]启发的基于PyTorch的开源时间序列Benchmark和工具箱。
目前仅专注于时间序列预测，后续可能会添加时间序列分类、异常检测等内容。
BasicTS为用户提供使用统一的、标准的Pipline（他是公平的，但可能不是最快的），来提供流行的TS模型的复现和公平对比，以启发新的创新。
BasicTS基于一个易用、强大的开源神经网络训练框架[EasyTorch](https://github.com/cnstark/easytorch) [2]开发。

## 1. Supported Models and Datasets

| Model\Dataset | METR-LA | PEMS-BAY | PEMS04 | PEMS08 | PEMS03 | PEMS07 | Other Datasets |
|:-------------:|:-------:|:--------:|:------:|:------:|:------:|:------:|:--------------:|
| AR            | ✅      | ✅       | ✅      | ✅     | 🕐      | 🕐     |                |
| VAR           | ✅      | ✅       | ✅      | ✅     | 🕐      | 🕐     |                |
| HI            | ✅      | ✅       | ✅      | ✅     | 🕐      | 🕐     |                |
| Graph WaveNet | ✅      | ✅       | ✅      | ✅     | 🕐      | 🕐     |                |
| DCRNN         | ✅      | ✅       | ✅      | ✅     | 🕐      | 🕐     |                |
| MTGNN         | 🕐      | 🕐       | 🕐      | 🕐     | 🕐      | 🕐     |                |
| GTS           | 🕐      | 🕐       | 🕐      | 🕐     | 🕐      | 🕐     |                |
| STGCN         | 🕐      | 🕐       | 🕐      | 🕐     | 🕐      | 🕐     |                |
| ASTGCN        | 🕐      | 🕐       | 🕐      | 🕐     | 🕐      | 🕐     |                |
| DGCRN         | 🕐      | 🕐       | 🕐      | 🕐     | 🕐      | 🕐     |                |
| AGCRN         | 🕐      | 🕐       | 🕐      | 🕐     | 🕐      | 🕐     |                |
| D2STGNN       | 🕐      | 🕐       | 🕐      | 🕐     | 🕐      | 🕐     |                |
| Other Models  |         |         |         |        |        |        |                |

For the results and more details about models and datasets, please refer to the [pdf](TODO)(🕐).

## 2. Dependencies

## 3. Usage

### 3.1 Data Preparation and Preprocessing

### 3.2 Run a Time Series Forecasting Model

### 3.3 Train a Custom Model

config&shape&m&runner

## 4. Codebase Designs and Conventions

## 5. Detailed Docs

- data preparation: data_preparation.md and [data_preparation_CN.md](docs/data_preparation_CN.md)

## 6. TODO

- [ ] : add 1.
- [ ] : add 3.3
- [ ] : add 4
- [ ] : add more docs
- [ ] : add README.md for dataset
- [ ] : upload datasets and training logs as well as the tensorboard to Google Drive and BaiDu Yun.
- [ ] : Add more model. Models that have official pytorch codes first.
  - [ ] RNN-based: DCRNN, GTS, DGCRN
  - [ ] CNN-based: MTGNN
  - [ ] Others: D2STGNN, STEP, AGCRN

## References

[1] Xintao Wang, Ke Yu, Kelvin C.K. Chan, Chao Dong and Chen Change Loy. BasicSR: Open Source Image and Video Restoration Toolbox. <https://github.com/xinntao/BasicSR>, 2018.\
[2] Yuhao Wang. EasyTorch. <https://github.com/cnstark/easytorch>, 2020.
