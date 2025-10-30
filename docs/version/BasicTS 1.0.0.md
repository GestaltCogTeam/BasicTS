# Version 1.0.0 beta

## 📣 Important
BasicTS version 1.0 is here! Welcome to experience the beta version first. BasicTS 1.0 has undergone a comprehensive upgrade focusing on **user-friendliness**, **multi-task support**, and **extensibility**.

## 🚀 New Features
- Optimized user experience, enabling model training or evaluation with just **three lines of Python code**
- Added **directly callable time series deep learning modules** (Transformer components, MLP, ...)
- Support for time series classification and imputation tasks
- Enhanced extensibility with new taskflow and callback mechanisms, allowing users to **modify workflows without changing the runner**
## 🐛 Bug Fixes
- Fixed the issue of potentially high GPU memory usage during model evaluation
- Fixed incorrect calculation of training and evaluation metrics
## 🔧 Upgrade Instructions
For BasicTS 1.0 beta version, there's no need to clone the repository anymore. Simply download `basicts-1.0-py3-none-any.whl` and execute:
```bash
pip install basicts-1.0-py3-none-any.whl
```
---
## 📣 重要消息
BasicTS 1.0 版本来了！欢迎抢先体验测试版。BasicTS 1.0 针对**用户友好**，**多任务支持**和**可扩展性**方面进行了全面升级。

## 🚀 新功能
- 优化了用户的使用体验，**三句Python**代码实现模型训练或评估
- 增加用户可以**直接调用的时序深度学习模块**（Transformer components, MLP, ...）
- 支持时间序列分类和插补任务
- 优化了可扩展性，新增taskflow和callback机制，用户**无需再修改runner**

## 🐛 修复问题
- 修复在GPU上评估模型显存可能占用过高的问题
- 修复训练、评估指标计算可能不正确的问题

## 🔧 升级说明
安装BasicTS v1.0 beta无需再clone仓库，只需下载`basicts-1.0-py3-none-any.whl`，然后执行：
```bash
pip install basicts-1.0-py3-none-any.whl
```