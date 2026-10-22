<div align="center">
  <img src="assets/Basic-TS-logo-for-white.png#gh-light-mode-only" height=200>
  <img src="assets/Basic-TS-logo-for-black.png#gh-dark-mode-only" height=200>
  <h3><b> 一个公平、可扩展的时间序列分析基准库和工具包 </b></h3>
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

🎉 [**快速上手**](./docs/getting_started_cn.md) **|**
💡 [**总体设计**](./docs/overall_design_cn.md)

📦 [**数据集 (Dataset)**](./docs/dataset_design_cn.md) **|**
🛠️ [**数据缩放 (Scaler)**](./docs/scaler_design_cn.md) **|**
🧠 [**模型约定 (Model)**](./docs/model_design_cn.md) **|**
📉 [**评估指标 (Metrics)**](./docs/metrics_design_cn.md)

🏃‍♂️ [**执行器 (Runner)**](./docs/runner_and_pipeline_cn.md) **|**
📜 [**配置文件 (Config)**](./docs/config_design_cn.md) **|**
📜 [**基线模型 (Baselines)**](./src/basicts/models/)

</div>

BasicTS (**Basic** **T**ime **S**eries) 是一个面向时间序列分析的基准库和工具箱，现已支持时空预测、长序列预测、分类、插补等多种任务与数据集，涵盖统计模型、机器学习模型、深度学习模型等多类算法，为开发和评估时间序列预测模型提供了理想的工具。你可以在[快速上手](./docs/getting_started_cn.md)找到详细的教程。

## 📢 最新动态

🎉 **更新（2025年10月）**：BasicTS 内置支持[**选择学习(NeurIPS'25)**](http://arxiv.org/abs/2510.25207)，一种有效缓解过拟合，增加模型性能和泛化性的训练策略。用户可以从[回调模块](./src/basicts/runners/callback/selective_learning.py)中导入并直接使用。[使用说明](https://github.com/GestaltCogTeam/selective-learning)

🎉 **更新（2025年10月）：BasicTS 1.0版本发布了！新特性：**
  - 🚀 **三行代码，快速上手**​​：pip install 安装，极简 API 设计，快速实现模型训练与评估。
  - 📦 **模块化组件，开箱即用**​​：提供 Transformer、MLP 等即插即用的组件，像搭积木一样构建自己的模型。
  - 🔄 **多任务支持**​​：支持时序预测、分类、插补等多个时序分析核心任务。
  - 🔧 **高可扩展架构**​​：基于 Taskflow 与 Callback 机制，无需修改 Runner 即可轻松定制。

🎉 **更新（2025年5月）：** BasicTS 现已支持使用 [**BLAST (KDD'25)**](https://arxiv.org/abs/2505.17871) 语料库训练通用预测模型（例如 **TimeMoE** 和 **ChronosBolt**）。BLAST 能够实现 **更快的收敛速度**、**显著降低计算成本**，并且即使在资源有限的情况下也能获得卓越性能。

如果你觉得这个项目对你有帮助，别忘了给个⭐Star支持一下，非常感谢！

> [!IMPORTANT]
> 如果本项目对您有用，请考虑引用下面的论文:
>
> ```LaTeX
> @article{shao2024exploring,
>  title={Exploring progress in multivariate time series forecasting: Comprehensive benchmarking and heterogeneity analysis},
>  author={Shao, Zezhi and Wang, Fei and Xu, Yongjun and Wei, Wei and Yu, Chengqing and Zhang, Zhao and Yao, Di and Sun, Tao and Jin, Guangyin and Cao, Xin and others},
>  journal={IEEE Transactions on Knowledge and Data Engineering},
>  year={2024},
>  volume={37},
>  number={1},
>  pages={291-305},
>  publisher={IEEE}
> }
> ```
>
> 🔥🔥🔥 ***该论文已被IEEE TKDE录用！你可以在这里[查看论文](https://arxiv.org/abs/2310.06119)。*** 🔥🔥🔥

## ✨ 主要功能亮点

BasicTS 一方面通过 **统一且标准化的流程**，为热门的深度学习模型提供了 **公平且全面** 的复现与对比平台。

另一方面，BasicTS 提供了用户 **友好且易于扩展** 的接口，帮助快速设计和评估新模型。用户只需定义模型结构，便可轻松完成基本操作。

### 公平的性能评估：

通过统一且全面的流程，用户能够公平且充分地对比不同模型在任意数据集上的性能表现。

### 使用 BasicTS 进行开发你可以：

<details>
  <summary><b>最简代码实现</b></summary>
用户只需实现关键部分如模型架构、数据预处理和后处理，即可构建自己的深度学习项目。
</details>

<details>
  <summary><b>基于配置文件控制一切</b></summary>
用户可以通过配置文件掌控流程中的所有细节，包括数据加载器的超参数、优化策略以及其他技巧（如课程学习）。
</details>

<details>
  <summary><b>支持所有设备</b></summary>
BasicTS 支持 CPU、GPU 以及分布式 GPU 训练（单节点多 GPU 和多节点），依托 EasyTorch 作为后端。用户只需通过设置参数即可使用这些功能，无需修改代码。
</details>

<details>
  <summary><b>保存训练日志</b></summary>
BasicTS 提供 `logging` 日志系统和 `Tensorboard` 支持，并统一封装接口，用户可以通过简便的接口调用来保存自定义的训练日志。
</details>

## 🚀 安装和快速上手

详细的安装步骤请参考 [快速上手](./docs/getting_started_cn.md) 教程。

## 📦 支持的模型

BasicTS 实现了丰富的基线模型，包括经典模型、时空预测模型、长序列预测模型、通用预测模型等。

这些模型的代码实现可在 [baselines](./baselines) 目录中找到。

下表中的代码链接（💻Code） 指向了相关论文的官方实现，感谢各位作者对代码的开源贡献！


<details open>
  <summary><h3>通用预测模型</h3></summary>

| 📊Baseline | 📝Title                                                                                                              | 📄Paper                                              | 💻Code                                                                                                                                 | 🏛Venue     | 🎯Task |
| :--------- | :------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------- | :----- |
| TimeMoE | Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts | [Link](https://openreview.net/forum?id=e1wDDFmlVu) | [Link](https://github.com/Time-MoE/Time-MoE) | ICLR'25 | UFM |
| ChronosBolt | Chronos: Learning the Language of Time Series | [Link](https://arxiv.org/abs/2403.07815) | [Link](https://github.com/amazon-science/chronos-forecasting) | TMLR'24 | UFM |
MOIRAI (inference) | Unified Training of Universal Time Series Forecasting Transformers  | [Link](https://arxiv.org/abs/2402.02592) | [Link](https://github.com/SalesforceAIResearch/uni2ts) | ICML'24 | UFM |
</details>

<details open>
  <summary><h3>时空预测</h3></summary>

| 📊Baseline | 📝Title                                                                                                              | 📄Paper                                              | 💻Code                                                                                                                                                                                        | 🏛Venue     | 🎯Task |
| :--------- | :------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------- | :----- |
| STDN       | Spatiotemporal-aware Trend-Seasonality Decomposition Network for Traffic Flow Forecasting                            | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/33247)   | [Link](https://github.com/roarer008/STDN)                                                                                                                                               | AAAI'25     | STF    |
| HimNet     | Heterogeneity-Informed Meta-Parameter Learning for Spatiotemporal Time Series Forecasting                            | [Link](https://arxiv.org/abs/2405.10800)                | [Link](https://github.com/XDZhelheim/HimNet)                                                                                                                                                     | SIGKDD'24   | STF    |
| DFDGCN     | Dynamic Frequency Domain Graph Convolutional Network for Traffic Forecasting                                         | [Link](https://arxiv.org/abs/2312.11933) | [Link](https://github.com/GestaltCogTeam/DFDGCN)                                                                                                                                                    | ICASSP'24   | STF    |
| STPGNN     | Spatio-Temporal Pivotal Graph Neural Networks for Trafﬁc Flow Forecasting                                            | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/28707) | [Link](https://github.com/Kongwy5689/STPGNN?tab=readme-ov-file)                                                                                                                           | AAAI'24     | STF    |
| BigST      | Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks          | [Link](https://dl.acm.org/doi/10.14778/3641204.3641217) | [Link](https://github.com/usail-hkust/BigST?tab=readme-ov-file)                                                                                                                                  | VLDB'24     | STF    |
| STDMAE     | Spatio-Temporal-Decoupled Masked Pre-training for Traffic Forecasting                                                | [Link](https://arxiv.org/abs/2312.00516)                | [Link](https://github.com/Jimmy-7664/STD-MAE)                                                                                                                                                    | IJCAI'24    | STF    |
| STWave     | When Spatio-Temporal Meet Wavelets: Disentangled Traffic Forecasting via Efficient Spectral Graph Attention Networks | [Link](https://ieeexplore.ieee.org/document/10184591)   | [Link](https://github.com/LMissher/STWave)                                                                                                                                                       | ICDE'23     | STF    |
| STAEformer | Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting                            | [Link](https://arxiv.org/abs/2308.10425)                | [Link](https://github.com/XDZhelheim/STAEformer)                                                                                                                                                 | CIKM'23     | STF    |
| MegaCRN    | Spatio-Temporal Meta-Graph Learning for Traffic Forecasting                                                          | [Link](https://aps.arxiv.org/abs/2212.05989)            | [Link](https://github.com/deepkashiwa20/MegaCRN)                                                                                                                                                 | AAAI'23     | STF    |
| DGCRN      | Dynamic Graph Convolutional Recurrent Network for Traffic Prediction: Benchmark and Solution                         | [Link](https://arxiv.org/abs/2104.14917)                | [Link](https://github.com/tsinghua-fib-lab/Traffic-Benchmark)                                                                                                                                    | ACM TKDD'23 | STF    |
| STID       | Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting                  | [Link](https://arxiv.org/abs/2208.05233)                | [Link](https://github.com/zezhishao/STID)                                                                                                                                                        | CIKM'22     | STF    |
| STEP       | Pretraining Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting                  | [Link](https://arxiv.org/abs/2206.09113)                | [Link](https://github.com/GestaltCogTeam/STEP?tab=readme-ov-file)                                                                                                                                | SIGKDD'22   | STF    |
| D2STGNN    | Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting                                      | [Link](https://arxiv.org/abs/2206.09112)                | [Link](https://github.com/zezhishao/D2STGNN)                                                                                                                                                     | VLDB'22     | STF    |
| STNorm     | Spatial and Temporal Normalization for Multi-variate Time Series Forecasting                                         | [Link](https://dl.acm.org/doi/10.1145/3447548.3467330)  | [Link](https://github.com/JLDeng/ST-Norm/blob/master/models/Wavenet.py)                                                                                                                          | SIGKDD'21   | STF    |
| STGODE     | Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting                                                     | [Link](https://arxiv.org/abs/2106.12931)                | [Link](https://github.com/square-coder/STGODE)                                                                                                                                                   | SIGKDD'21   | STF    |
| GTS        | Discrete Graph Structure Learning for Forecasting Multiple Time Series                                               | [Link](https://arxiv.org/abs/2101.06861)                | [Link](https://github.com/chaoshangcs/GTS)                                                                                                                                                       | ICLR'21     | STF    |
| StemGNN    | Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting                                      | [Link](https://arxiv.org/abs/2103.07719)                | [Link](https://github.com/microsoft/StemGNN)                                                                                                                                                     | NeurIPS'20  | STF    |
| MTGNN      | Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks                                 | [Link](https://arxiv.org/abs/2005.11650)                | [Link](https://github.com/nnzhan/MTGNN)                                                                                                                                                          | SIGKDD'20   | STF    |
| AGCRN      | Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting                                               | [Link](https://arxiv.org/abs/2007.02842)                | [Link](https://github.com/LeiBAI/AGCRN)                                                                                                                                                          | NeurIPS'20  | STF    |
| GWNet      | Graph WaveNet for Deep Spatial-Temporal Graph Modeling                                                               | [Link](https://arxiv.org/abs/1906.00121)                | [Link](https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py)                                                                                                                             | IJCAI'19    | STF    |
| STGCN      | Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting                      | [Link](https://arxiv.org/abs/1709.04875)                | [Link](https://github.com/VeritasYin/STGCN_IJCAI-18)                                                                                                                                             | IJCAI'18    | STF    |
| DCRNN      | Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting                                    | [Link](https://arxiv.org/abs/1707.01926)                | [Link1](https://github.com/chnsh/DCRNN_PyTorch/blob/pytorch_scratch/model/pytorch/dcrnn_cell.py), [Link2](https://github.com/chnsh/DCRNN_PyTorch/blob/pytorch_scratch/model/pytorch/dcrnn_model.py) | ICLR'18     | STF    |

</details>

<details open>
  <summary><h3>Long-Term Time Series Forecasting</h3></summary>

| 📊Baseline    | 📝Title                                                                                                  | 📄Paper                                                | 💻Code                                                                        | 🏛Venue    | 🎯Task |
| :------------ | :------------------------------------------------------------------------------------------------------- | :----------------------------------------------------- | :---------------------------------------------------------------------------- | :--------- | :----- |
| OLinear       | OLinear: A Linear Model for Time Series Forecasting in Orthogonally Transformed Domain                      | [Link](https://papers.neurips.cc/paper_files/paper/2025/hash/571c7e164fb1ffbcf2f84a63784451ec-Abstract-Conference.html) | [Link](https://github.com/jackyue1994/OLinear)                                    | NeurIPS'25 | LTSF   |
| TQNet         | Temporal Query Network for Efficient Multivariate Time Series Forecasting                                  | [Link](https://proceedings.mlr.press/v267/lin25e.html)     | [Link](https://github.com/ACAT-SCUT/TQNet)                                        | ICML'25    | LTSF   |
| S-D-Mamba         | Is Mamba Effective for Time Series Forecasting?                           | [Link](https://arxiv.org/abs/2403.11144v3)                  | [Link](https://github.com/wzhwzhwzh0921/S-D-Mamba)                                        | NeuroComputing'24 | LTSF   |
| Bi-Mamba         | Bi-Mamba+: Bidirectional Mamba for Time Series Forecasting                           | [Link](https://arxiv.org/abs/2404.15772)                  | [Link](https://github.com/Leopold2333/Bi-Mamba4TS)                                        | arXiv'24 | LTSF   |
| ModernTCN         | ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis                           | [Link](https://openreview.net/forum?id=vpJMJerXHU)                  | [Link](https://github.com/luodhhh/ModernTCN)                                        | ICLR'24 | LTSF   |
| TimeXer         | TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables                           | [Link](https://arxiv.org/abs/2402.19072)                  | [Link](https://github.com/thuml/TimeXer)                                        | NeurIPS'24 | LTSF   |
| CARD         | CARD: Channel Aligned Robust Blend Transformer for Time Series Forecasting                           | [Link](https://arxiv.org/abs/2305.12095)                  | [Link](https://github.com/wxie9/CARD)                                        | ICLR'24 | LTSF   |
| SOFTS         | SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion                           | [Link](https://arxiv.org/pdf/2404.14197)                  | [Link](https://github.com/Secilia-Cxy/SOFTS)                                        | NeurIPS'24 | LTSF   |
| CATS          | Are Self-Attentions Effective for Time Series Forecasting?                                               | [Link](https://arxiv.org/pdf/2405.16877)                  | [Link](https://github.com/dongbeank/CATS)                                        | NeurIPS'24 | LTSF   |
| Sumba         | Structured Matrix Basis for Multivariate Time Series Forecasting with Interpretable Dynamics             | [Link](https://xiucheng.org/assets/pdfs/nips24-sumba.pdf) | [Link](https://github.com/chenxiaodanhit/Sumba/)                                 | NeurIPS'24 | LTSF   |
| GLAFF         | Rethinking the Power of Timestamps for Robust Time Series Forecasting: A Global-Local Fusion Perspective | [Link](https://arxiv.org/pdf/2409.18696)                  | [Link](https://github.com/ForestsKing/GLAFF)                                     | NeurIPS'24 | LTSF   |
| CycleNet      | CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns Forecasting               | [Link](https://arxiv.org/pdf/2409.18479)                  | [Link](https://github.com/ACAT-SCUT/CycleNet)                                    | NeurIPS'24 | LTSF   |
| Fredformer    | Fredformer: Frequency Debiased Transformer for Time Series Forecasting                                   | [Link](https://arxiv.org/pdf/2406.09009)                  | [Link](https://github.com/chenzRG/Fredformer)                                    | KDD'24     | LTSF   |
| UMixer        | An Unet-Mixer Architecture with Stationarity Correction for Time Series Forecasting                      | [Link](https://arxiv.org/abs/2401.02236)                  | [Link](https://github.com/XiangMa-Shaun/U-Mixer)                                 | AAAI'24    | LTSF   |
| TimeMixer     | Decomposable Multiscale Mixing for Time Series Forecasting                                               | [Link](https://arxiv.org/html/2405.14616v1)               | [Link](https://github.com/kwuking/TimeMixer)                                     | ICLR'24    | LTSF   |
| Time-LLM      | Time-LLM: Time Series Forecasting by Reprogramming Large Language Models                                 | [Link](https://arxiv.org/abs/2310.01728)                  | [Link](https://github.com/KimMeen/Time-LLM)                                      | ICLR'24    | LTSF   |
| SparseTSF     | Modeling LTSF with 1k Parameters                                                                         | [Link](https://arxiv.org/abs/2405.00946)                  | [Link](https://github.com/lss-1138/SparseTSF)                                    | ICML'24    | LTSF   |
| iTrainsformer | Inverted Transformers Are Effective for Time Series Forecasting                                          | [Link](https://arxiv.org/abs/2310.06625)                  | [Link](https://github.com/thuml/iTransformer)                                    | ICLR'24    | LTSF   |
| Koopa         | Learning Non-stationary Time Series Dynamics with Koopman Predictors                                     | [Link](https://arxiv.org/abs/2305.18803)                  | [Link](https://github.com/thuml/Koopa)                                           | NeurIPS'24 | LTSF   |
| CrossGNN      | CrossGNN: Confronting Noisy Multivariate Time Series Via Cross Interaction Refinement                    | [Link](https://openreview.net/pdf?id=xOzlW2vUYc)          | [Link](https://github.com/hqh0728/CrossGNN)                                      | NeurIPS'23 | LTSF   |
| NLinear       | Are Transformers Effective for Time Series Forecasting?                                                  | [Link](https://arxiv.org/abs/2205.13504)                  | [Link](https://github.com/cure-lab/DLinear)                                      | AAAI'23    | LTSF   |
| Crossformer   | Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting                | [Link](https://openreview.net/forum?id=vSVLM2j9eie)       | [Link](https://github.com/Thinklab-SJTU/Crossformer)                             | ICLR'23    | LTSF   |
| DLinear       | Are Transformers Effective for Time Series Forecasting?                                                  | [Link](https://arxiv.org/abs/2205.13504)                  | [Link](https://github.com/cure-lab/DLinear)                                      | AAAI'23    | LTSF   |
| DSformer      | A Double Sampling Transformer for Multivariate Time Series Long-term Prediction                          | [Link](https://arxiv.org/abs/2308.03274)                  | [Link](https://github.com/ChengqingYu/DSformer)                                  | CIKM'23    | LTSF   |
| SegRNN        | Segment Recurrent Neural Network for Long-Term Time Series Forecasting                                   | [Link](https://arxiv.org/abs/2308.11200)                  | [Link](https://github.com/lss-1138/SegRNN)                                       | arXiv      | LTSF   |
| MTS-Mixers    | Multivariate Time Series Forecasting via Factorized Temporal and Channel Mixing                          | [Link](https://arxiv.org/abs/2302.04501)                  | [Link](https://github.com/plumprc/MTS-Mixers)                                    | arXiv      | LTSF   |
| LightTS       | Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP                               | [Link](https://arxiv.org/abs/2207.01186)                  | [Link](https://github.com/thuml/Time-Series-Library/blob/main/models/LightTS.py) | arXiv      | LTSF   |
| ETSformer     | Exponential Smoothing Transformers for Time-series Forecasting                                           | [Link](https://arxiv.org/abs/2202.01381)                  | [Link](https://github.com/salesforce/ETSformer)                                  | arXiv      | LTSF   |
| NHiTS         | Neural Hierarchical Interpolation for Time Series Forecasting                                            | [Link](https://arxiv.org/abs/2201.12886)                  | [Link](https://github.com/cchallu/n-hits)                                        | AAAI'23    | LTSF   |
| PatchTST      | A Time Series is Worth 64 Words: Long-term Forecasting with Transformers                                 | [Link](https://arxiv.org/abs/2211.14730)                  | [Link](https://github.com/yuqinie98/PatchTST)                                    | ICLR'23    | LTSF   |
| TiDE          | Long-term Forecasting with TiDE: Time-series Dense Encoder                                               | [Link](https://arxiv.org/abs/2304.08424)                  | [Link](https://github.com/lich99/TiDE)                                           | TMLR'23    | LTSF   |
| S4         | Efficiently Modeling Long Sequences with Structured State Spaces                           | [Link](https://openreview.net/pdf?id=uYLFoz1vlAC)                  | [Link](https://github.com/state-spaces/s4)                                        | ICLR'22 | LTSF   |
| TimesNet      | Temporal 2D-Variation Modeling for General Time Series Analysis                                          | [Link](https://openreview.net/pdf?id=ju_Uqw384Oq)         | [Link](https://github.com/thuml/TimesNet)                                        | ICLR'23    | LTSF   |
| Triformer     | Triangular, Variable-Specific Attentions for Long Sequence Multivariate Time Series Forecasting          | [Link](https://arxiv.org/abs/2204.13767)                  | [Link](https://github.com/razvanc92/triformer)                                   | IJCAI'22   | LTSF   |
| NSformer      | Exploring the Stationarity in Time Series Forecasting                                                    | [Link](https://arxiv.org/abs/2205.14415)                  | [Link](https://github.com/thuml/Nonstationary_Transformers)                      | NeurIPS'22 | LTSF   |
| FiLM          | Frequency improved Legendre Memory Model for LTSF                                                        | [Link](https://arxiv.org/abs/2205.08897)                  | [Link](https://github.com/tianzhou2011/FiLM)                                     | NeurIPS'22 | LTSF   |
| FEDformer     | Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting                               | [Link](https://arxiv.org/abs/2201.12740v3)                | [Link](https://github.com/MAZiqing/FEDformer)                                    | ICML'22    | LTSF   |
| Pyraformer    | Low complexity pyramidal Attention For Long-range Time Series Modeling and Forecasting                   | [Link](https://openreview.net/forum?id=0EXmFzUn5I)        | [Link](https://github.com/ant-research/Pyraformer)                               | ICLR'22    | LTSF   |
| HI            | Historical Inertia: A Powerful Baseline for Long Sequence Time-series Forecasting                        | [Link](https://arxiv.org/abs/2103.16349)                  | None                                                                          | CIKM'21    | LTSF   |
| Autoformer    | Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting                        | [Link](https://arxiv.org/abs/2106.13008)                  | [Link](https://github.com/thuml/Autoformer)                                      | NeurIPS'21 | LTSF   |
| Informer      | Beyond Efficient Transformer for Long Sequence Time-Series Forecasting                                   | [Link](https://arxiv.org/abs/2012.07436)                  | [Link](https://github.com/zhouhaoyi/Informer2020)                                | AAAI'21    | LTSF   |

</details>

<details open>
  <summary><h3>预测归一化方法</h3></summary>

| 📊Baseline | 📝Title | 📄Paper | 💻Code | 🏛Venue | 🎯Task |
| :--------- | :------ | :------ | :----- | :------ | :----- |
| APT | APT: Affine Prototype-Timestamp for Time Series Forecasting Under Distribution Shift | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/38542) | [Link](https://github.com/blisky-li/APT) | AAAI'26 | LTSF |
| FAN | Frequency Adaptive Normalization For Non-stationary Time Series Forecasting | [Link](https://proceedings.neurips.cc/paper_files/paper/2024/file/37c6d0bc4d2917dcbea693b18504bd87-Paper-Conference.pdf) | [Link](https://github.com/wayne155/FAN) | NeurIPS'24 | LTSF |
| SAN | Adaptive Normalization for Non-stationary Time Series Forecasting: A Temporal Slice Perspective | [Link](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2e19dab94882bc95ed094c4399cfda02-Abstract-Conference.html) | [Link](https://github.com/ustc-time-series/SAN) | NeurIPS'23 | LTSF |
| Dish-TS | Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting | [Link](https://ojs.aaai.org/index.php/AAAI/article/view/25914) | [Link](https://github.com/weifantt/Dish-TS) | AAAI'23 | LTSF |
| RevIN | Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift | [Link](https://openreview.net/forum?id=cGDAkQo1C0p) | [Link](https://github.com/ts-kim/RevIN) | ICLR'22 | LTSF |
| DAIN | Deep Adaptive Input Normalization for Time Series Forecasting | [Link](https://doi.org/10.1109/TNNLS.2019.2944933) | [Link](https://github.com/passalis/dain) | IEEE TNNLS'20 | LTSF |

</details>

<details open>
  <summary><h3>其他方法</h3></summary>

| 📊Baseline | 📝Title                                                                   | 📄Paper                                                                                        | 💻Code                                                                                                                                                 | 🏛Venue             | 🎯Task                                |
| :--------- | :------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------ | :------------------------------------ |
| CatBoost   | Catboost: unbiased boosting with categorical features              | [Link](https://proceedings.neurips.cc/paper_files/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf) | [Link](https://github.com/catboost/catboost)                                                                                                             | NeurIPS'18          | Machine Learning                      |
| LightGBM   | LightGBM: A Highly Efficient Gradient Boosting Decision Tree              | [Link](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf) | [Link](https://github.com/microsoft/LightGBM)                                                                                                             | NeurIPS'17          | Machine Learning                      |
| NBeats     | Neural basis expansion analysis for interpretable time series forecasting | [Link](https://arxiv.org/abs/1905.10437)                                                          | [Link1](https://github.com/ServiceNow/N-BEATS), [Link2](https://github.com/philipperemy/n-beats)                                                             | ICLR'19             | Deep Time Series Forecasting          |
| DeepAR     | Probabilistic Forecasting with Autoregressive Recurrent Networks          | [Link](https://arxiv.org/abs/1704.04110)                                                          | [Link1](https://github.com/jingw2/demand_forecast), [Link2](https://github.com/husnejahan/DeepAR-pytorch), [Link3](https://github.com/arrigonialberto86/deepar) | Int. J. Forecast'20 | Probabilistic Time Series Forecasting |
| WaveNet    | WaveNet: A Generative Model for Raw Audio.                                | [Link](https://arxiv.org/abs/1609.03499)                                                          | [Link 1](https://github.com/JLDeng/ST-Norm/blob/master/models/Wavenet.py), [Link 2](https://github.com/huyouare/WaveNet-Theano)                              | arXiv               | Audio                                 |
| AR   | VII. On a method of investigating periodicities disturbed series, with special reference to Wolfer's sunspot numbers              | [Link](https://royalsocietypublishing.org/doi/abs/10.1098/rsta.1927.0007) | [Link](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html)                                                                                                             | 1927          | Local Forecasting                      |
| MA   | On periodicity in series of related terms              | [Link](https://royalsocietypublishing.org/doi/10.1098/rspa.1931.0069) | [Link](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html)                                                                                                             | 1931          | Local Forecasting                      |
| ARMA   | Some recent advances in forecasting and control              | [Link](https://www.jstor.org/stable/2985674) | [Link](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html)                                                                                                             | Applied Statistics'1968         | Local Forecasting                      |
| ARIMA   | Forecasting with exponential smoothing: the state space approach             | [Link](https://link.springer.com/chapter/10.1007/978-3-540-71918-2_12) | [Link](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html)                                                                                                             | 2008         | Local Forecasting                      |
| SARIMA   | Forecasting with exponential smoothing: the state space approach              | [Link](https://link.springer.com/chapter/10.1007/978-3-540-71918-2_12) | [Link](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html)                                                                                                             | 2008         | Local Forecasting                      |
| ARCH   | Conditional heteroscedasticity in time series of stock returns: Evidence and forecasts              | [Link](https://www.jstor.org/stable/2353123) | [Link](https://pypi.org/project/arch/)                                                                                                             | Journal of business'1989         | Local Forecasting                      |
| GARCH   | Conditional heteroscedasticity in time series of stock returns: Evidence and forecasts               | [Link](https://www.jstor.org/stable/2353123) | [Link](https://pypi.org/project/arch/)                                                                                                             | Journal of business'1989         | Local Forecasting                      |
| ETS   | The holt-winters forecasting procedure              | [Link](https://www.jstor.org/stable/2347162) | [Link](https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html)                                                                                                             | Applied Statistics'1978          | Local Forecasting                    |
| SES   | The holt-winters forecasting procedure              | [Link](https://www.jstor.org/stable/2347162) | [Link](https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.SimpleExpSmoothing.html)                                                                                                             | Applied Statistics'1978          | Local Forecasting                    |
| SVR   | Support vector regression machines              | [Link](https://proceedings.neurips.cc/paper_files/paper/1996/file/d38901788c533e8286cb6400b40b386d-Paper.pdf) | [Link](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)                                                                                                             | NeurIPS'1996          | Machine Learning                      |
| PolySVR  | A training algorithm for optimal margin classifiers              | [Link](https://dl.acm.org/doi/abs/10.1145/130385.130401)| [Link](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)                                                                                                             | COLT'1992          | Machine Learning                      |


</details>

## 📦 支持的数据集

BasicTS 支持多种类型的数据集，涵盖时空预测、长序列预测及大规模数据集。

<details open>
  <summary><h3>时空预测</h3></summary>

| 🏷️Name | 🌐Domain      | 📏Length | 📊Time Series Count | 🔄Graph | ⏱️Freq. (m) | 🎯Task |
| :------- | :------------ | -------: | ------------------: | :------ | ------------: | :----- |
| METR-LA  | Traffic Speed |    34272 |                 207 | True    |             5 | STF    |
| PEMS-BAY | Traffic Speed |    52116 |                 325 | True    |             5 | STF    |
| PEMS03   | Traffic Flow  |    26208 |                 358 | True    |             5 | STF    |
| PEMS04   | Traffic Flow  |    16992 |                 307 | True    |             5 | STF    |
| PEMS07   | Traffic Flow  |    28224 |                 883 | True    |             5 | STF    |
| PEMS08   | Traffic Flow  |    17856 |                 170 | True    |             5 | STF    |

</details>

<details open>
  <summary><h3>长序列预测</h3></summary>

| 🏷️Name          | 🌐Domain                            | 📏Length | 📊Time Series Count | 🔄Graph | ⏱️Freq. (m) | 🎯Task |
| :---------------- | :---------------------------------- | -------: | ------------------: | :------ | ------------: | :----- |
| BeijingAirQuality | Beijing Air Quality                 |    36000 |                   7 | False   |            60 | LTSF   |
| ETTh1             | Electricity Transformer Temperature |    14400 |                   7 | False   |            60 | LTSF   |
| ETTh2             | Electricity Transformer Temperature |    14400 |                   7 | False   |            60 | LTSF   |
| ETTm1             | Electricity Transformer Temperature |    57600 |                   7 | False   |            15 | LTSF   |
| ETTm2             | Electricity Transformer Temperature |    57600 |                   7 | False   |            15 | LTSF   |
| Electricity       | Electricity Consumption             |    26304 |                 321 | False   |            60 | LTSF   |
| ExchangeRate      | Exchange Rate                       |     7588 |                   8 | False   |          1440 | LTSF   |
| Illness           | Ilness Data                         |      966 |                   7 | False   |         10080 | LTSF   |
| Traffic           | Road Occupancy Rates                |    17544 |                 862 | False   |            60 | LTSF   |
| Weather           | Weather                             |    52696 |                  21 | False   |            10 | LTSF   |

</details>

<details open>
  <summary><h3>大规模数据集</h3></summary>

| 🏷️Name | 🌐Domain     | 📏Length | 📊Time Series Count | 🔄Graph | ⏱️Freq. (m) | 🎯Task      |
| :------- | :----------- | -------: | ------------------: | :------ | ------------: | :---------- |
| CA       | Traffic Flow |    35040 |                8600 | True    |            15 | Large Scale |
| GBA      | Traffic Flow |    35040 |                2352 | True    |            15 | Large Scale |
| GLA      | Traffic Flow |    35040 |                3834 | True    |            15 | Large Scale |
| SD       | Traffic Flow |    35040 |                 716 | True    |            15 | Large Scale |

</details>

<details open>
  <summary><h3>Pre-training Corpus</h3></summary>

| 🏷️Name | 🌐Domain     | 📏Length | 📊Time Series Count | 🔄Graph | ⏱️Freq. | 🎯Task      |
| :------- | :----------- | -------: | ------------------: | :------ | ------------: | :---------- |
|   [BLAST](https://github.com/GestaltCogTeam/BasicTS/blob/master/docs/training_with_BLAST_cn.md)  |  Multiple |     4096 |      20000000       | False  |   Multiple  |     UFM     |

## 📉 主要结果

请参阅论文 *[多变量时间序列预测进展探索：全面基准评测和异质性分析](https://arxiv.org/pdf/2310.06119.pdf)*。

## ✨ 贡献者

感谢这些优秀的贡献者们 ([表情符号指南](https://allcontributors.org/docs/en/emoji-key))：

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/zezhishao"><img src="https://avatars.githubusercontent.com/u/33691477?v=4?s=100" width="100px;" alt="S22"/><br /><sub><b>S22</b></sub></a><br /><a href="#maintenance-zezhishao" title="Maintenance">🚧</a> <a href="https://github.com/GestaltCogTeam/BasicTS/commits?author=zezhishao" title="Code">💻</a> <a href="https://github.com/GestaltCogTeam/BasicTS/issues?q=author%3Azezhishao" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/finleywang"><img src="https://avatars.githubusercontent.com/u/5022944?v=4?s=100" width="100px;" alt="finleywang"/><br /><sub><b>finleywang</b></sub></a><br /><a href="#mentoring-finleywang" title="Mentoring">🧑‍🏫</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/blisky-li"><img src="https://avatars.githubusercontent.com/u/66107694?v=4?s=100" width="100px;" alt="blisky-li"/><br /><sub><b>blisky-li</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/commits?author=blisky-li" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/LMissher"><img src="https://avatars.githubusercontent.com/u/37818979?v=4?s=100" width="100px;" alt="LMissher"/><br /><sub><b>LMissher</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/commits?author=LMissher" title="Code">💻</a> <a href="https://github.com/GestaltCogTeam/BasicTS/issues?q=author%3ALMissher" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/cnstark"><img src="https://avatars.githubusercontent.com/u/45590791?v=4?s=100" width="100px;" alt="CNStark"/><br /><sub><b>CNStark</b></sub></a><br /><a href="#infra-cnstark" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Azusa-Yuan"><img src="https://avatars.githubusercontent.com/u/61765965?v=4?s=100" width="100px;" alt="Azusa"/><br /><sub><b>Azusa</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/issues?q=author%3AAzusa-Yuan" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ywoelker"><img src="https://avatars.githubusercontent.com/u/94364022?v=4?s=100" width="100px;" alt="Yannick Wölker"/><br /><sub><b>Yannick Wölker</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/issues?q=author%3Aywoelker" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/hlhang9527"><img src="https://avatars.githubusercontent.com/u/77621248?v=4?s=100" width="100px;" alt="hlhang9527"/><br /><sub><b>hlhang9527</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/issues?q=author%3Ahlhang9527" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ChengqingYu"><img src="https://avatars.githubusercontent.com/u/114470704?v=4?s=100" width="100px;" alt="Chengqing Yu"/><br /><sub><b>Chengqing Yu</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/commits?author=ChengqingYu" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Reborn14"><img src="https://avatars.githubusercontent.com/u/74488779?v=4?s=100" width="100px;" alt="Reborn14"/><br /><sub><b>Reborn14</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/commits?author=Reborn14" title="Documentation">📖</a> <a href="https://github.com/GestaltCogTeam/BasicTS/commits?author=Reborn14" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/TensorPulse"><img src="https://avatars.githubusercontent.com/u/94754159?v=4?s=100" width="100px;" alt="TensorPulse"/><br /><sub><b>TensorPulse</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/issues?q=author%3ATensorPulse" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/superarthurlx"><img src="https://avatars.githubusercontent.com/u/40826115?v=4?s=100" width="100px;" alt="superarthurlx"/><br /><sub><b>superarthurlx</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/commits?author=superarthurlx" title="Code">💻</a> <a href="https://github.com/GestaltCogTeam/BasicTS/issues?q=author%3Asuperarthurlx" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/yisongfu"><img src="https://avatars.githubusercontent.com/u/139831104?v=4?s=100" width="100px;" alt="Yisong Fu"/><br /><sub><b>Yisong Fu</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/commits?author=yisongfu" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/DiamonJoy"><img src="https://avatars.githubusercontent.com/u/11327242?v=4?s=100" width="100px;" alt="Xubin"/><br /><sub><b>Xubin</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/commits?author=DiamonJoy" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/duyifanict"><img src="https://avatars.githubusercontent.com/u/171222220?v=4?s=100" width="100px;" alt="DU YIFAN"/><br /><sub><b>DU YIFAN</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/commits?author=duyifanict" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

此项目遵循 [all-contributors](https://github.com/all-contributors/all-contributors) 规范。欢迎任何形式的贡献！

<!-- ## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GestaltCogTeam/BasicTS&type=Date)](https://star-history.com/#GestaltCogTeam/BasicTS&Date) -->

## 🔗 致谢

BasicTS 是基于 [EasyTorch](https://github.com/cnstark/easytorch) 开发的，这是一个易于使用且功能强大的开源神经网络训练框架。

## 📧 联系我们

欢迎加入我们的官方社区，在这里您可以获取更多技术支持，与志同道合的伙伴交流，共同探讨领域内的最新研究进展。

官方微信群:

![wechat](assets/BasicTS-wechat-cn.jpg)

官方Discord频道: [点击加入我们的 Discord 社区](https://discord.gg/UWFP7b3b7H)
