<div align="center">
  <img src="assets/Basic-TS-logo-for-white.png#gh-light-mode-only" height=200>
  <img src="assets/Basic-TS-logo-for-black.png#gh-dark-mode-only" height=200>
  <h3><b> A Fair and Scalable Time Series Forecasting Benchmark and Toolkit. </b></h3>
</div>

<div align="center">

[**English**](./README.md) **|** 
[**简体中文**](./README_CN.md)

</div>

---

<div align="center">

[![EasyTorch](https://img.shields.io/badge/Developing%20with-EasyTorch-2077ff.svg)](https://github.com/cnstark/easytorch)
[![LICENSE](https://img.shields.io/github/license/zezhishao/BasicTS.svg)](https://github.com/zezhishao/BasicTS/blob/master/LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10.0-orange)](https://pytorch.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-orange)](https://pytorch.org/)
[![python lint](https://github.com/zezhishao/BasicTS/actions/workflows/pylint.yml/badge.svg)](https://github.com/zezhishao/BasicTS/blob/master/.github/workflows/pylint.yml)

</div>

<div align="center">

🎉 [**Getting Started**](./tutorial/getting_started.md) **|** 
💡 [**Overall Design**](./tutorial/overall_design.md)

📦 [**Dataset**](./tutorial/dataset_design.md) **|** 
🛠️ [**Scaler**](./tutorial/scaler_design.md) **|** 
🧠 [**Model**](./tutorial/model_design.md) **|** 
📉 [**Metrics**](./tutorial/metrics_design.md) **|** 
🏃‍♂️ [**Runner**](./tutorial/runner_design.md) **|** 
📜 [**Config**](./tutorial/config_design.md.md) **|** 
📜 [**Baselines**](./baselines/)

</div>

$\text{BasicTS}^{+}$ (**Basic** **T**ime **S**eries) is a benchmark library and toolkit designed for time series forecasting. It now supports a wide range of tasks and datasets, including spatial-temporal forecasting and long-term time series forecasting. It covers various types of algorithms such as statistical models, machine learning models, and deep learning models, making it an ideal tool for developing and evaluating time series forecasting models.

If you find this project helpful, please don't forget to give it a ⭐ Star to show your support. Thank you!

On one hand, BasicTS provides a **unified and standardized pipeline**, offering a **fair and comprehensive** platform for reproducing and comparing popular models.

On the other hand, BasicTS offers a **user-friendly and easily extensible** interface, enabling quick design and evaluation of new models. Users can simply define their model structure and easily perform basic operations.

You can find detailed tutorials in [Getting Started](./tutorial/getting_started.md). Additionally, we are collecting **ToDo** and **HowTo** items. If you need more features (e.g., additional datasets or benchmark models) or tutorials, feel free to open an issue or leave a comment [here](https://github.com/zezhishao/BasicTS/issues/95).


> [!IMPORTANT]  
> If you find this repository helpful for your work, please consider citing the following benchmarking paper:
> ```LaTeX
> @article{shao2023exploring,
>    title={Exploring Progress in Multivariate Time Series Forecasting: Comprehensive Benchmarking and Heterogeneity Analysis},
>    author={Shao, Zezhi and Wang, Fei and Xu, Yongjun and Wei, Wei and Yu, Chengqing and Zhang, Zhao and Yao, Di and Jin, Guangyin and Cao, Xin and Cong, Gao and others},
>    journal={arXiv preprint arXiv:2310.06119},
>    year={2023}
>  }
>  ```
> 🔥🔥🔥 ***The paper has been accepted by IEEE TKDE! You can check it out [here](https://arxiv.org/abs/2310.06119).***  🔥🔥🔥


## ✨ Highlighted Features

### Fair Performance Review

Users can compare the performance of different models on arbitrary datasets fairly and exhaustively based on a unified and comprehensive pipeline.

### Developing with BasicTS

<details>
  <summary><b>Minimum Code</b></summary>
Users only need to implement key codes such as model architecture and data pre/post-processing to build their own deep learning projects.
</details>

<details>
  <summary><b>Everything Based on Config</b></summary>
Users can control all the details of the pipeline through a config file, such as the hyperparameter of dataloaders, optimization, and other tricks (*e.g.*, curriculum learning). 
</details>

<details>
  <summary><b>Support All Devices</b></summary>
BasicTS supports CPU, GPU and GPU distributed training (both single node multiple GPUs and multiple nodes) thanks to using EasyTorch as the backend. Users can use it by setting parameters without modifying any code.
</details>

<details>
  <summary><b>Save Training Log</b></summary>
Support `logging` log system and `Tensorboard`, and encapsulate it as a unified interface, users can save customized training logs by calling simple interfaces.
</details>

## 🚀 Installation and Quick Start

For detailed instructions, please refer to the [Getting Started](./tutorial/getting_started.md) tutorial.

## 📦 Supported Baselines

BasicTS implements a wealth of models, including ***classic models***, ***spatial-temporal forecasting*** models, and ***long-term time series forecasting*** model:

You can find the implementation of these models in the [baselines](./baselines) directory.

The code links (💻Code) in the table below point to the official implementations from these papers. Many thanks to the authors for open-sourcing their work!

<details open>
  <summary><h3>Spatial-Temporal Forecasting</h3></summary>


| 📊Baseline   | 📝Title                                                                                                              | 📄Paper                                                | 💻Code                                                                                                                                                                                              | 🏛Venue      | 🎯Task   |
|:-------------|:---------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------|:---------|
| BigST        | Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks          | [Link](https://dl.acm.org/doi/10.14778/3641204.3641217)  | [Link](https://github.com/usail-hkust/BigST?tab=readme-ov-file)                              | VLDB'24     | STF      |
| STDMAE       | Spatio-Temporal-Decoupled Masked Pre-training for Traffic Forecasting                                                | [Link](https://arxiv.org/abs/2312.00516)  | [Link](https://github.com/Jimmy-7664/STD-MAE)                                                                                                                                                          | IJCAI'24     | STF      |
| STWave       | When Spatio-Temporal Meet Wavelets: Disentangled Traffic Forecasting via Efficient Spectral Graph Attention Networks | [Link](https://ieeexplore.ieee.org/document/10184591)  | [Link](https://github.com/LMissher/STWave)                                                                                                                                                          | ICDE'23     | STF      |
| STAEformer   | Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting                            | [Link](https://arxiv.org/abs/2308.10425)               | [Link](https://github.com/XDZhelheim/STAEformer)                                                                                                                                                    | CIKM'23     | STF      |
| MegaCRN      | Spatio-Temporal Meta-Graph Learning for Traffic Forecasting                                                          | [Link](https://aps.arxiv.org/abs/2212.05989)           | [Link](https://github.com/deepkashiwa20/MegaCRN)                                                                                                                                                    | AAAI'23     | STF      |
| DGCRN        | Dynamic Graph Convolutional Recurrent Network for Traffic Prediction: Benchmark and Solution                         | [Link](https://arxiv.org/abs/2104.14917)               | [Link](https://github.com/tsinghua-fib-lab/Traffic-Benchmark)                                                                                                                                       | ACM TKDD'23 | STF      |
| STID         | Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting                  | [Link](https://arxiv.org/abs/2208.05233)               | [Link](https://github.com/zezhishao/STID)                                                                                                                                                           | CIKM'22     | STF      |
| STEP         | Pretraining Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting                  | [Link](https://arxiv.org/abs/2206.09113)               | [Link](https://github.com/GestaltCogTeam/STEP?tab=readme-ov-file)                                                                                                                                   | SIGKDD'22   | STF      |
| D2STGNN      | Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting                                      | [Link](https://arxiv.org/abs/2206.09112)               | [Link](https://github.com/zezhishao/D2STGNN)                                                                                                                                                        | VLDB'22     | STF      |
| STNorm       | Spatial and Temporal Normalization for Multi-variate Time Series Forecasting                                         | [Link](https://dl.acm.org/doi/10.1145/3447548.3467330) | [Link](https://github.com/JLDeng/ST-Norm/blob/master/models/Wavenet.py)                                                                                                                             | SIGKDD'21   | STF      |
| STGODE       | Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting                                                     | [Link](https://arxiv.org/abs/2106.12931)               | [Link](https://github.com/square-coder/STGODE)                                                                                                                                                      | SIGKDD'21   | STF      |
| GTS          | Discrete Graph Structure Learning for Forecasting Multiple Time Series                                               | [Link](https://arxiv.org/abs/2101.06861)               | [Link](https://github.com/chaoshangcs/GTS)                                                                                                                                                          | ICLR'21     | STF      |
| StemGNN      | Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting                                      | [Link](https://arxiv.org/abs/2103.07719)               | [Link](https://github.com/microsoft/StemGNN)                                                                                                                                                        | NeurIPS'20  | STF      |
| MTGNN        | Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks                                 | [Link](https://arxiv.org/abs/2005.11650)               | [Link](https://github.com/nnzhan/MTGNN)                                                                                                                                                             | SIGKDD'20   | STF      |
| AGCRN        | Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting                                               | [Link](https://arxiv.org/abs/2007.02842)               | [Link](https://github.com/LeiBAI/AGCRN)                                                                                                                                                             | NeurIPS'20  | STF      |
| GWNet        | Graph WaveNet for Deep Spatial-Temporal Graph Modeling                                                               | [Link](https://arxiv.org/abs/1906.00121)               | [Link](https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py)                                                                                                                                | IJCAI'19    | STF      |
| STGCN        | Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting                      | [Link](https://arxiv.org/abs/1709.04875)               | [Link](https://github.com/VeritasYin/STGCN_IJCAI-18)                                                                                                                                                | IJCAI'18    | STF      |
| DCRNN        | Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting                                    | [Link](https://arxiv.org/abs/1707.01926)               | [Link1](https://github.com/chnsh/DCRNN_PyTorch/blob/pytorch_scratch/model/pytorch/dcrnn_cell.py), [Link2](https://github.com/chnsh/DCRNN_PyTorch/blob/pytorch_scratch/model/pytorch/dcrnn_model.py) | ICLR'18     | STF      |
</details>

<details open>
  <summary><h3>Long-Term Time Series Forecasting</h3></summary>


| 📊Baseline    | 📝Title                                                                                         | 📄Paper                                             | 💻Code                                                                           | 🏛Venue     | 🎯Task   |
|:--------------|:------------------------------------------------------------------------------------------------|:----------------------------------------------------|:---------------------------------------------------------------------------------|:-----------|:---------|
| Fredformer    | Fredformer: Frequency Debiased Transformer for Time Series Forecasting                          | [Link]( https://arxiv.org/pdf/2406.09009)            | [Link](https://github.com/chenzRG/Fredformer)                                       | KDD'24      | LTSF     |
| UMixer        | An Unet-Mixer Architecture with Stationarity Correction for Time Series Forecasting             | [Link](https://arxiv.org/abs/2401.02236)            | [Link](https://github.com/XiangMa-Shaun/U-Mixer)                                 | AAAI'24    | LTSF     |
| TimeMixer     | Decomposable Multiscale Mixing for Time Series Forecasting                                      | [Link](https://arxiv.org/html/2405.14616v1)         | [Link](https://github.com/kwuking/TimeMixer)                                     | ICLR'24    | LTSF     |
| Time-LLM      | Time-LLM: Time Series Forecasting by Reprogramming Large Language Models                        | [Link](https://arxiv.org/abs/2310.01728)            | [Link](https://github.com/KimMeen/Time-LLM)                                      | ICLR'24    | LTSF     |
| SparseTSF     | Modeling LTSF with 1k Parameters                                                                | [Link](https://arxiv.org/abs/2405.00946)            | [Link](https://github.com/lss-1138/SparseTSF)                                    | ICML'24    | LTSF     |
| iTrainsformer | Inverted Transformers Are Effective for Time Series Forecasting                                 | [Link](https://arxiv.org/abs/2310.06625)            | [Link](https://github.com/thuml/iTransformer)                                    | ICLR'24    | LTSF     |
| Koopa         | Learning Non-stationary Time Series Dynamics with Koopman Predictors                            | [Link](https://arxiv.org/abs/2305.18803)            | [Link](https://github.com/thuml/Koopa)                                           | NeurIPS'24 | LTSF     |
| CrossGNN      | CrossGNN: Confronting Noisy Multivariate Time Series Via Cross Interaction Refinement           | [Link](https://openreview.net/pdf?id=xOzlW2vUYc)    | [Link](https://github.com/hqh0728/CrossGNN)                                      | NeurIPS'23 | LTSF     |
| NLinear       | Are Transformers Effective for Time Series Forecasting?                                         | [Link](https://arxiv.org/abs/2205.13504)            | [Link](https://github.com/cure-lab/DLinear)                                      | AAAI'23    | LTSF     |
| Crossformer   | Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting       | [Link](https://openreview.net/forum?id=vSVLM2j9eie) | [Link](https://github.com/Thinklab-SJTU/Crossformer)                             | ICLR'23    | LTSF     |
| DLinear       | Are Transformers Effective for Time Series Forecasting?                                         | [Link](https://arxiv.org/abs/2205.13504)            | [Link](https://github.com/cure-lab/DLinear)                                      | AAAI'23    | LTSF     |
| DSformer      | A Double Sampling Transformer for Multivariate Time Series Long-term Prediction                 | [Link](https://arxiv.org/abs/2308.03274)            | [Link](https://github.com/ChengqingYu/DSformer)                                  | CIKM'23    | LTSF     |
| SegRNN        | Segment Recurrent Neural Network for Long-Term Time Series Forecasting                          | [Link](https://arxiv.org/abs/2308.11200)            | [Link](https://github.com/lss-1138/SegRNN)                                       | arXiv      | LTSF     |
| MTS-Mixers    | Multivariate Time Series Forecasting via Factorized Temporal and Channel Mixing                 | [Link](https://arxiv.org/abs/2302.04501)            | [Link](https://github.com/plumprc/MTS-Mixers)                                    | arXiv      | LTSF     |
| LightTS       | Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP                      | [Link](https://arxiv.org/abs/2207.01186)            | [Link](https://github.com/thuml/Time-Series-Library/blob/main/models/LightTS.py) | arXiv      | LTSF     |
| ETSformer     | Exponential Smoothing Transformers for Time-series Forecasting                                  | [Link](https://arxiv.org/abs/2202.01381)            | [Link](https://github.com/salesforce/ETSformer)                                  | arXiv      | LTSF     |
| NHiTS         | Neural Hierarchical Interpolation for Time Series Forecasting                                   | [Link](https://arxiv.org/abs/2201.12886)            | [Link](https://github.com/cchallu/n-hits)                                        | AAAI'23    | LTSF     |
| PatchTST        | A Time Series is Worth 64 Words: Long-term Forecasting with Transformers                      | [Link](https://arxiv.org/abs/2211.14730)            | [Link](https://github.com/yuqinie98/PatchTST)                                  | ICLR'23    | LTSF     |
| TiDE          | Long-term Forecasting with TiDE: Time-series Dense Encoder                                      | [Link](https://arxiv.org/abs/2304.08424)            | [Link](https://github.com/lich99/TiDE)                                           | TMLR'23    | LTSF     |
| TimesNet      | Temporal 2D-Variation Modeling for General Time Series Analysis                                 | [Link](https://openreview.net/pdf?id=ju_Uqw384Oq)   | [Link](https://github.com/thuml/TimesNet)                                        | ICLR'23    | LTSF     |
| Triformer     | Triangular, Variable-Specific Attentions for Long Sequence Multivariate Time Series Forecasting | [Link](https://arxiv.org/abs/2204.13767)            | [Link](https://github.com/razvanc92/triformer)                                   | IJCAI'22   | LTSF     |
| NSformer      | Exploring the Stationarity in Time Series Forecasting                                           | [Link](https://arxiv.org/abs/2205.14415)            | [Link](https://github.com/thuml/Nonstationary_Transformers)                      | NeurIPS'22 | LTSF     |
| FiLM          | Frequency improved Legendre Memory Model for LTSF                                               | [Link](https://arxiv.org/abs/2205.08897)            | [Link](https://github.com/tianzhou2011/FiLM)                                     | NeurIPS'22 | LTSF     |
| FEDformer     | Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting                      | [Link](https://arxiv.org/abs/2201.12740v3)          | [Link](https://github.com/MAZiqing/FEDformer)                                    | ICML'22    | LTSF     |
| Pyraformer    | Low complexity pyramidal Attention For Long-range Time Series Modeling and Forecasting          | [Link](https://openreview.net/forum?id=0EXmFzUn5I)  | [Link](https://github.com/ant-research/Pyraformer)                               | ICLR'22    | LTSF     |
| HI           | Historical Inertia: A Powerful Baseline for Long Sequence Time-series Forecasting                | [Link](https://arxiv.org/abs/2103.16349)            | None                                                                             | CIKM'21    | LTSF     |
| Autoformer    | Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting               | [Link](https://arxiv.org/abs/2106.13008)            | [Link](https://github.com/thuml/Autoformer)                                      | NeurIPS'21 | LTSF     |
| Informer      | Beyond Efficient Transformer for Long Sequence Time-Series Forecasting                          | [Link](https://arxiv.org/abs/2012.07436)            | [Link](https://github.com/zhouhaoyi/Informer2020)                                | AAAI'21    | LTSF     |
</details>


<details open>
  <summary><h3>Others</h3></summary>


| 📊Baseline   | 📝Title                                                                           | 📄Paper                                                                                           | 💻Code                                                                                                                                                          | 🏛Venue              | 🎯Task                          |
|:-------------|:----------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------|:--------------------------------|
| LightGBM     | LightGBM: A Highly Efficient Gradient Boosting Decision Tree                      | [Link](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf) | [Link](https://github.com/microsoft/LightGBM)                                                                                                                   | NeurIPS'17          | Machine Learning |
| NBeats       | Neural basis expansion analysis for interpretable time series forecasting         | [Link](https://arxiv.org/abs/1905.10437)                                                          | [Link1](https://github.com/ServiceNow/N-BEATS), [Link2](https://github.com/philipperemy/n-beats)                                                                | ICLR'19             | Deep Time Series Forecasting |
| DeepAR       | Probabilistic Forecasting with Autoregressive Recurrent Networks                  | [Link](https://arxiv.org/abs/1704.04110)                                                          | [Link1](https://github.com/jingw2/demand_forecast), [Link2](https://github.com/husnejahan/DeepAR-pytorch), [Link3](https://github.com/arrigonialberto86/deepar) | Int. J. Forecast'20 | Probabilistic Time Series Forecasting |
| WaveNet      | WaveNet: A Generative Model for Raw Audio.                                        | [Link](https://arxiv.org/abs/1609.03499)                                                          | [Link 1](https://github.com/JLDeng/ST-Norm/blob/master/models/Wavenet.py), [Link 2](https://github.com/huyouare/WaveNet-Theano)                                 | arXiv               | Audio |
</details>


## 📦 Supported Datasets

BasicTS support a variety of datasets, including ***spatial-temporal forecasting***, ***long-term time series forecasting***, and ***large-scale*** datasets.

<details open>
  <summary><h3>Spatial-Temporal Forecasting</h3></summary>

| 🏷️Name   | 🌐Domain     | 📏Length |   📊Time Series Count | 🔄Graph |   ⏱️Freq. (m) | 🎯Task      |
|:---------|:--------------|-----------:|----------------------:|:--------------------|--------------------:|:---------|
| METR-LA  | Traffic Speed |      34272 |                   207 | True                |                   5 | STF      |
| PEMS-BAY | Traffic Speed |      52116 |                   325 | True                |                   5 | STF      |
| PEMS03   | Traffic Flow  |      26208 |                   358 | True                |                   5 | STF      |
| PEMS04   | Traffic Flow  |      16992 |                   307 | True                |                   5 | STF      |
| PEMS07   | Traffic Flow  |      28224 |                   883 | True                |                   5 | STF      |
| PEMS08   | Traffic Flow  |      17856 |                   170 | True                |                   5 | STF      |
</details>

<details open>
  <summary><h3>Long-Term Time Series Forecasting</h3></summary>

| 🏷️Name   | 🌐Domain     | 📏Length |   📊Time Series Count | 🔄Graph |   ⏱️Freq. (m) | 🎯Task      |
|:------------------|:------------------------------------|-----------:|----------------------:|:--------------------|--------------------:|:---------|
| BeijingAirQuality | Beijing Air Quality                 |      36000 |                     7 | False               |                  60 | LTSF     |
| ETTh1             | Electricity Transformer Temperature |      14400 |                     7 | False               |                  60 | LTSF     |
| ETTh2             | Electricity Transformer Temperature |      14400 |                     7 | False               |                  60 | LTSF     |
| ETTm1             | Electricity Transformer Temperature |      57600 |                     7 | False               |                  15 | LTSF     |
| ETTm2             | Electricity Transformer Temperature |      57600 |                     7 | False               |                  15 | LTSF     |
| Electricity       | Electricity Consumption             |      26304 |                   321 | False               |                  60 | LTSF     |
| ExchangeRate      | Exchange Rate                       |       7588 |                     8 | False               |                1440 | LTSF     |
| Illness           | Ilness Data                         |        966 |                     7 | False               |               10080 | LTSF     |
| Traffic           | Road Occupancy Rates                |      17544 |                   862 | False               |                  60 | LTSF     |
| Weather           | Weather                             |      52696 |                    21 | False               |                  10 | LTSF     |
</details>

<details open>
  <summary><h3>Large Scale Dataset</h3></summary>

| 🏷️Name   | 🌐Domain     | 📏Length |   📊Time Series Count | 🔄Graph |   ⏱️Freq. (m) | 🎯Task      |
|:---------|:-------------|-----------:|----------------------:|:--------------------|--------------------:|:------------|
| CA       | Traffic Flow |      35040 |                  8600 | True                |                  15 | Large Scale |
| GBA      | Traffic Flow |      35040 |                  2352 | True                |                  15 | Large Scale |
| GLA      | Traffic Flow |      35040 |                  3834 | True                |                  15 | Large Scale |
| SD       | Traffic Flow |      35040 |                   716 | True                |                  15 | Large Scale |

</details>

## 📉 Main Results

See the paper *[Exploring Progress in Multivariate Time Series Forecasting:
Comprehensive Benchmarking and Heterogeneity Analysis](https://arxiv.org/pdf/2310.06119.pdf).*

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/zezhishao"><img src="https://avatars.githubusercontent.com/u/33691477?v=4?s=100" width="100px;" alt="S22"/><br /><sub><b>S22</b></sub></a><br /><a href="#maintenance-zezhishao" title="Maintenance">🚧</a> <a href="https://github.com/GestaltCogTeam/BasicTS/commits?author=zezhishao" title="Code">💻</a> <a href="https://github.com/GestaltCogTeam/BasicTS/issues?q=author%3Azezhishao" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/blisky-li"><img src="https://avatars.githubusercontent.com/u/66107694?v=4?s=100" width="100px;" alt="blisky-li"/><br /><sub><b>blisky-li</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/commits?author=blisky-li" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/LMissher"><img src="https://avatars.githubusercontent.com/u/37818979?v=4?s=100" width="100px;" alt="LMissher"/><br /><sub><b>LMissher</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/commits?author=LMissher" title="Code">💻</a> <a href="https://github.com/GestaltCogTeam/BasicTS/issues?q=author%3ALMissher" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/cnstark"><img src="https://avatars.githubusercontent.com/u/45590791?v=4?s=100" width="100px;" alt="CNStark"/><br /><sub><b>CNStark</b></sub></a><br /><a href="#infra-cnstark" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Azusa-Yuan"><img src="https://avatars.githubusercontent.com/u/61765965?v=4?s=100" width="100px;" alt="Azusa"/><br /><sub><b>Azusa</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/issues?q=author%3AAzusa-Yuan" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ywoelker"><img src="https://avatars.githubusercontent.com/u/94364022?v=4?s=100" width="100px;" alt="Yannick Wölker"/><br /><sub><b>Yannick Wölker</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/issues?q=author%3Aywoelker" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/hlhang9527"><img src="https://avatars.githubusercontent.com/u/77621248?v=4?s=100" width="100px;" alt="hlhang9527"/><br /><sub><b>hlhang9527</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/issues?q=author%3Ahlhang9527" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ChengqingYu"><img src="https://avatars.githubusercontent.com/u/114470704?v=4?s=100" width="100px;" alt="Chengqing Yu"/><br /><sub><b>Chengqing Yu</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/commits?author=ChengqingYu" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Reborn14"><img src="https://avatars.githubusercontent.com/u/74488779?v=4?s=100" width="100px;" alt="Reborn14"/><br /><sub><b>Reborn14</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/commits?author=Reborn14" title="Documentation">📖</a> <a href="https://github.com/GestaltCogTeam/BasicTS/commits?author=Reborn14" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/TensorPulse"><img src="https://avatars.githubusercontent.com/u/94754159?v=4?s=100" width="100px;" alt="TensorPulse"/><br /><sub><b>TensorPulse</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/issues?q=author%3ATensorPulse" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/superarthurlx"><img src="https://avatars.githubusercontent.com/u/40826115?v=4?s=100" width="100px;" alt="superarthurlx"/><br /><sub><b>superarthurlx</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/commits?author=superarthurlx" title="Code">💻</a> <a href="https://github.com/GestaltCogTeam/BasicTS/issues?q=author%3Asuperarthurlx" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/yisongfu"><img src="https://avatars.githubusercontent.com/u/139831104?v=4?s=100" width="100px;" alt="Yisong Fu"/><br /><sub><b>Yisong Fu</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/commits?author=yisongfu" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GestaltCogTeam/BasicTS&type=Date)](https://star-history.com/#GestaltCogTeam/BasicTS&Date)

## 🔗 Acknowledgement

BasicTS is developed based on [EasyTorch](https://github.com/cnstark/easytorch), an easy-to-use and powerful open-source neural network training framework.
