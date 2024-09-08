# 🏃‍♂️ 执行器 (Runner)

## 💿 概述

执行器 是 BasicTS 的核心组件，负责管理整个训练和评估过程。它将数据集、数据缩放器、模型、评估指标和配置文件等各个子组件集成在一起，构建一个公平且可扩展的训练和评估流程。执行器 提供了多项高级功能，包括但不限于：

- 提前停止
- 课程学习
- 梯度裁剪
- 模型自动保存
- 多 GPU 训练
- 持久化日志记录

执行器 可用于训练和评估模型。

## ⚡️ 训练流程

使用 执行器 的典型训练流程如下：

```python
# 初始化
runner = Runner(config)  # 包含缩放器、模型、评估指标、损失、优化器等

# 训练
runner.train(config)
```

`runner.train` 方法的工作原理如下：

```python
def train(config):
    init_training(config)  # 初始化训练/验证/测试数据加载器
    for epoch in train_epochs:
        on_epoch_start(epoch)
        for data in train_dataloader:
            loss = train_iters(data)
            optimize(loss)  # 包含反向传播、学习率调度、梯度裁剪等
        on_epoch_end(epoch)
    on_training_end(config)
```

### Hook 函数

执行器 提供了一些 Hook 函数，例如 `on_epoch_start`、`on_epoch_end` 和 `on_training_end`，允许用户实现自定义逻辑。例如，`on_epoch_end` 可以用于评估验证集和测试集并保存中间模型，而 `on_training_end` 通常用于最终评估并保存最终模型和结果。

### 训练迭代

`runner.train_iters` 的流程如下：

```python
def train_iters(data):
    data = runner.preprocessing(data)  # 归一化数据
    forward_return = runner.forward(data)  # 前向传递
    forward_return = runner.postprocessing(forward_return)  # 反归一化结果
    loss = runner.loss(forward_return)  # 计算损失
    metrics = runner.metrics(forward_return)  # 计算评估指标
    return loss
```

默认情况下，`runner.preprocessing` 只归一化 `inputs` 和 `target`。如果数据集中还有其他参数需要归一化，您需要自定义 `runner.preprocessing` 函数。同样地，`runner.postprocessing` 默认会反归一化 `inputs`、`target` 和 `prediction`，如果更多参数需要反归一化，您也需要自定义 `runner.postprocessing` 函数。

`runner.forward` 函数处理模型输入并将模型输出打包成一个包含 `prediction`、`inputs`、`target` 和其他用于计算评估指标的参数的字典。

## ✨ 评估流程

当评估模型性能时，流程通常如下：

```python
# 初始化
runner = Runner(config)  # 包含缩放器、模型、评估指标、损失、优化器等

# 加载模型权重
runner.load_model(checkpoint)

# 评估
runner.test_pipeline(config)
```

`runner.test_pipeline` 方法的工作原理如下：

```python
def test_pipeline(config):
    init_testing(config)  # 初始化测试数据加载器
    all_data = []
    for data in test_dataloader:
        data = runner.preprocessing(data)  # 归一化数据
        forward_return = runner.forward(data)  # 前向传递
        forward_return = runner.postprocessing(forward_return)  # 反归一化结果
        all_data.append(forward_return)
    all_data = concatenate(all_data)
    metrics = runner.metrics(all_data)  # 计算评估指标
    save(forward_return, metrics)  # 可选
```

## 🛠️ 自定义 执行器

BasicTS 提供了 [`SimpleTimeSeriesForecastingRunner`](../basicts/runners/runner_zoo/simple_tsf_runner.py) 类，处理大多数使用场景。

如果有更具体的需求，您可以扩展 [`SimpleTimeSeriesForecastingRunner`](../basicts/runners/runner_zoo/simple_tsf_runner.py) 或 [`BaseTimeSeriesForecastingRunner`](../basicts/runners/base_tsf_runner.py) 类，来实现 `test`、`forward`、`preprocessing`、`postprocessing` 和 `train_iters` 等函数。

## 🧑‍💻 进一步探索

- **🎉 [快速上手](./getting_started_cn.md)**
- **💡 [了解 BasicTS 的设计理念](./overall_design_cn.md)**
- **📦 [探索数据集设计并自定义数据集](./dataset_design_cn.md)**
- **🛠️ [了解数据缩放器设计并创建自定义缩放器](./scaler_design_cn.md)**
- **🧠 [深入了解模型设计并构建自定义模型](./model_design_cn.md)**
- **📉 [了解评估指标设计并开发自定义损失函数与评估指标](./metrics_design_cn.md)**
- **🏃‍♂️ [掌握执行器设计并创建自定义执行器](./runner_design_cn.md)**
- **📜 [解析配置文件设计并自定义配置](./config_design_cn.md)**
- **🔍 [探索多种基线模型](../baselines/)**
