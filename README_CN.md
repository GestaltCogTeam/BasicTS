<div align="center">
  <img src="assets/Basic-TS-logo-for-white.png#gh-light-mode-only" height=200>
  <img src="assets/Basic-TS-logo-for-black.png#gh-dark-mode-only" height=200>
  <h3><b> 一个公平、可扩展的时间序列预测基准库和工具包 </b></h3>
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

🎉 [**快速上手**](./tutorial/getting_started_cn.md) **|** 
💡 [**总体设计**](./tutorial/overall_design_cn.md)

📦 [**数据集 (Dataset)**](./tutorial/dataset_design_cn.md) **|** 
🛠️ [**数据缩放 (Scaler)**](./tutorial/scaler_design_cn.md) **|** 
🧠 [**模型约定 (Model)**](./tutorial/model_design_cn.md) **|** 
📉 [**评估指标 (Metrics)**](./tutorial/metrics_design_cn.md)

🏃‍♂️ [**执行器 (Runner)**](./tutorial/runner_design_cn.md) **|** 
📜 [**配置文件 (Config)**](./tutorial/config_design_cn.md) **|** 
📜 [**基线模型 (Baselines)**](./baselines/)

</div>

$\text{BasicTS}^{+}$ (**Basic** **T**ime **S**eries) 是一个面向时间序列预测的基准库和工具箱，现已支持时空预测、长序列预测等多种任务与数据集，涵盖统计模型、机器学习模型、深度学习模型等多类算法，为开发和评估时间序列预测模型提供了理想的工具。

如果你觉得这个项目对你有帮助，别忘了给个⭐Star支持一下，非常感谢！

BasicTS 一方面通过 **统一且标准化的流程**，为热门的深度学习模型提供了 **公平且全面** 的复现与对比平台。

另一方面，BasicTS 提供了用户 **友好且易于扩展** 的接口，帮助快速设计和评估新模型。用户只需定义模型结构，便可轻松完成基本操作。

你可以在[快速上手](./tutorial/getting_started_cn.md)找到详细的教程。另外，我们正在收集 **ToDo** 和 **HowTo**，如果您需要更多功能（例如：更多数据集或基准模型）或教程，欢迎提出 issue 或在[此处](https://github.com/zezhishao/BasicTS/issues/95)留言。

> [!IMPORTANT]  
> 如果本项目对您有用，请引用如下文献:
> ```LaTeX
> @article{shao2023exploring,
>    title={Exploring Progress in Multivariate Time Series Forecasting: Comprehensive Benchmarking and Heterogeneity Analysis},
>    author={Shao, Zezhi and Wang, Fei and Xu, Yongjun and Wei, Wei and Yu, Chengqing and Zhang, Zhao and Yao, Di and Jin, Guangyin and Cao, Xin and Cong, Gao and others},
>    journal={arXiv preprint arXiv:2310.06119},
>    year={2023}
>  }
>  ```

## ✨ 主要功能亮点

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

## 📦 内置数据集和基准模型

### 数据集

BasicTS 支持多种类型的数据集，涵盖时空预测、长序列预测及大规模数据集，例如：

- METR-LA, PEMS-BAY, PEMS03, PEMS04, PEMS07, PEMS08
- ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Exchange Rate, Weather, Traffic, Illness, Beijing Air Quality
- SD, GLA, GBA, CA
- ...

### 基准模型

BasicTS 实现了多种经典模型、时空预测模型和长序列预测模型，例如：

- HI, DeepAR, LightGBM, ...
- DCRNN, Graph WaveNet, MTGNN, STID, D2STGNN, STEP, DGCRN, DGCRN, STNorm, AGCRN, GTS, StemGNN, MegaCRN, STGCN, STWave, STAEformer, GMSDR, ...
- Informer, Autoformer, FEDformer, Pyraformer, DLinear, NLinear, Triformer, Crossformer, ...

## 🚀 安装和快速上手

详细的安装步骤请参考 [快速上手](./tutorial/getting_started_cn.md) 教程。

## 📉 主要结果

请参阅论文 *[多变量时间序列预测进展探索：全面基准评测和异质性分析](https://arxiv.org/pdf/2310.06119.pdf)*。

## 贡献者 ✨

感谢这些优秀的贡献者们 ([表情符号指南](https://allcontributors.org/docs/en/emoji-key))：

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/zezhishao"><img src="https://avatars.githubusercontent.com/u/33691477?v=4?s=100" width="100px;" alt="S22"/><br /><sub><b>S22</b></sub></a><br /><a href="#maintenance-zezhishao" title="Maintenance">🚧</a> <a href="https://github.com/GestaltCogTeam/BasicTS/commits?author=zezhishao" title="Code">💻</a> <a href="https://github.com/GestaltCogTeam/BasicTS/issues?q=author%3Azezhishao" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/LMissher"><img src="https://avatars.githubusercontent.com/u/37818979?v=4?s=100" width="100px;" alt="LMissher"/><br /><sub><b>LMissher</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/commits?author=LMissher" title="Code">💻</a> <a href="https://github.com/GestaltCogTeam/BasicTS/issues?q=author%3ALMissher" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/cnstark"><img src="https://avatars.githubusercontent.com/u/45590791?v=4?s=100" width="100px;" alt="CNStark"/><br /><sub><b>CNStark</b></sub></a><br /><a href="#infra-cnstark" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Azusa-Yuan"><img src="https://avatars.githubusercontent.com/u/61765965?v=4?s=100" width="100px;" alt="Azusa"/><br /><sub><b>Azusa</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/issues?q=author%3AAzusa-Yuan" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ywoelker"><img src="https://avatars.githubusercontent.com/u/94364022?v=4?s=100" width="100px;" alt="Yannick Wölker"/><br /><sub><b>Yannick Wölker</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/issues?q=author%3Aywoelker" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/hlhang9527"><img src="https://avatars.githubusercontent.com/u/77621248?v=4?s=100" width="100px;" alt="hlhang9527"/><br /><sub><b>hlhang9527</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/issues?q=author%3Ahlhang9527" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ChengqingYu"><img src="https://avatars.githubusercontent.com/u/114470704?v=4?s=100" width="100px;" alt="Chengqing Yu"/><br /><sub><b>Chengqing Yu</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/commits?author=ChengqingYu" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Reborn14"><img src="https://avatars.githubusercontent.com/u/74488779?v=4?s=100" width="100px;" alt="Reborn14"/><br /><sub><b>Reborn14</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/commits?author=Reborn14" title="Documentation">📖</a> <a href="https://github.com/GestaltCogTeam/BasicTS/commits?author=Reborn14" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/blisky-li"><img src="https://avatars.githubusercontent.com/u/66107694?v=4?s=100" width="100px;" alt="blisky-li"/><br /><sub><b>blisky-li</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/commits?author=blisky-li" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/TensorPulse"><img src="https://avatars.githubusercontent.com/u/94754159?v=4?s=100" width="100px;" alt="TensorPulse"/><br /><sub><b>TensorPulse</b></sub></a><br /><a href="https://github.com/GestaltCogTeam/BasicTS/issues?q=author%3ATensorPulse" title="Bug reports">🐛</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

此项目遵循 [all-contributors](https://github.com/all-contributors/all-contributors) 规范。欢迎任何形式的贡献！

## 🔗 致谢

BasicTS 是基于 [EasyTorch](https://github.com/cnstark/easytorch) 开发的，这是一个易于使用且功能强大的开源神经网络训练框架。