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

$\text{BasicTS}^{+}$ (**Basic** **T**ime **S**eries) is a benchmark library and toolkit designed for time series forecasting. It now supports a wide range of tasks and datasets, including spatial-temporal forecasting and long time series forecasting. It covers various types of algorithms such as statistical models, machine learning models, and deep learning models, making it an ideal tool for developing and evaluating time series forecasting models.

If you find this project helpful, please don't forget to give it a ⭐ Star to show your support. Thank you!

On one hand, BasicTS provides a **unified and standardized pipeline**, offering a **fair and comprehensive** platform for reproducing and comparing popular models.

On the other hand, BasicTS offers a **user-friendly and easily extensible** interface, enabling quick design and evaluation of new models. Users can simply define their model structure and easily perform basic operations.

You can find detailed tutorials in [Getting Started](./tutorial/getting_started.md). Additionally, we are collecting **ToDo** and **HowTo** items. If you need more features (e.g., additional datasets or benchmark models) or tutorials, feel free to open an issue or leave a comment [here](https://github.com/zezhishao/BasicTS/issues/95).


> [!IMPORTANT]  
> If you find this repository useful for your work, please cite it as such:
> ```LaTeX
> @article{shao2023exploring,
>    title={Exploring Progress in Multivariate Time Series Forecasting: Comprehensive Benchmarking and Heterogeneity Analysis},
>    author={Shao, Zezhi and Wang, Fei and Xu, Yongjun and Wei, Wei and Yu, Chengqing and Zhang, Zhao and Yao, Di and Jin, Guangyin and Cao, Xin and Cong, Gao and others},
>    journal={arXiv preprint arXiv:2310.06119},
>    year={2023}
>  }
>  ```

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

## 📦 Built-in Datasets and Baselines

### Datasets

BasicTS support a variety of datasets, including spatial-temporal forecasting, long time-series forecasting, and large-scale datasets, e.g.,

- METR-LA, PEMS-BAY, PEMS03, PEMS04, PEMS07, PEMS08
- ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Exchange Rate, Weather, Traffic, Illness, Beijing Air Quality
- SD, GLA, GBA, CA
- ...

### Baselines

BasicTS implements a wealth of models, including classic models, spatial-temporal forecasting models, and long time-series forecasting model, e.g.,
- HI, DeepAR, LightGBM, ...
- DCRNN, Graph WaveNet, MTGNN, STID, D2STGNN, STEP, DGCRN, DGCRN, STNorm, AGCRN, GTS, StemGNN, MegaCRN, STGCN, STWave, STAEformer, GMSDR, ...
- Informer, Autoformer, FEDformer, Pyraformer, DLinear, NLinear, Triformer, Crossformer, ...

## 🚀 Installation and Quick Start

For detailed instructions, please refer to the [Getting Started](./tutorial/getting_started.md) tutorial.

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
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## 🔗 Acknowledgement

BasicTS is developed based on [EasyTorch](https://github.com/cnstark/easytorch), an easy-to-use and powerful open-source neural network training framework.
