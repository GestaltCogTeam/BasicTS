# 🛠️ 数据缩放器设计 (Scaler)

## 🧐 什么是数据缩放器，为什么需要它？

数据缩放器（简称缩放器）是一个用于处理数据归一化和反归一化的类。在时间序列分析中，原始数据通常具有显著的尺度差异。因此，模型（尤其是深度学习模型）通常不会直接在原始数据上进行操作。相反，缩放器会将数据归一化到一个特定范围内，使其更适合建模。在计算损失函数或评估指标时，也可能将数据反归一化回原始尺度，以确保比较的准确性。

这使得缩放器成为时间序列分析工作流程中的重要组件。

## 👾 缩放器如何初始化及何时起作用？

缩放器与其他组件一起在执行器中初始化。

例如，Z-Score 缩放器会读取原始数据，并基于训练数据计算均值和标准差。

缩放器在从数据集中提取数据后起作用。数据首先由缩放器归一化，然后传递给模型进行训练。模型处理完数据后，缩放器会将输出反归一化，然后再传递给执行器进行损失计算和指标评估。

> [!IMPORTANT]  
> 在许多时间序列分析研究中，归一化通常在数据预处理中进行，这也是早期 BasicTS 版本的做法。然而，这种方式的可扩展性较差。诸如更改输入/输出长度、应用不同的归一化方法（例如对每个时间序列单独归一化），或更改训练/验证/测试集的比例等调整，都会要求重新预处理数据。为了解决这个问题，BasicTS 采用了“即时归一化”的方式，每次提取数据时都会进行归一化处理。

```python
# 在 runner 中
for data in dataloader:
    data = scaler.transform(data)
    forward_return = forward(data)
    forward_return = scaler.inverse_transform(forward_return)
```

## 🧑‍🔧 如何选择或自定义缩放器

BasicTS 提供了几种常见的缩放器，例如 Z-Score 和 Min-Max 缩放器。您可以通过在配置文件中设置 `CFG.SCALER.TYPE` 来轻松切换缩放器。

如果您需要自定义缩放器，可以扩展 `basicts.scaler.BaseScaler` 类，并实现 `transform` 和 `inverse_transform` 方法。或者，您也可以选择不继承该类，但仍然需要实现这两个方法。

## 🧑‍💻 进一步探索

- **🎉 [快速上手](./getting_started_cn.md)**
- **💡 [了解 BasicTS 的设计理念](./overall_design_cn.md)**
- **📦 [探索数据集设计并自定义数据集](./dataset_design_cn.md)**
- **🛠️ [了解数据缩放器设计并创建自定义缩放器](./scaler_design_cn.md)**
- **🧠 [深入了解模型设计并构建自定义模型](./model_design_cn.md)**
- **📉 [了解评估指标设计并开发自定义损失函数与评估指标](./metrics_design_cn.md)**
- **🏃‍♂️ [掌握执行器设计并创建自定义执行器](./runner_design_cn.md)**
- **📜 [解析配置文件设计并自定义配置](./config_design_cn.md)**
- **🎯 [探索使用BasicTS进行时间序列分类](./time_series_classification_cn.md)**
- **🔍 [探索多种基线模型](../baselines/)**
