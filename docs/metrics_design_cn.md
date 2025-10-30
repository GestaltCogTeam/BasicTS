# 📉 评估指标设计

## 接口规范

评估指标是评估模型性能的重要组成部分。在 BasicTS 中，评估指标是接受模型预测值、真实值及其他参数作为输入并返回标量值以评估模型性能的函数。

一个定义良好的评估指标函数应包含以下参数：
- **prediction**: 模型的预测值
- **targets**: 实际的真实值
- **targets_mask**: 可选参数，用于指定在哪些点上计算损失（一般用于掩码缺失值）。

`prediction` 和 `target` 是必需参数，而 `targets_mask` 是可选参数，但强烈建议采纳，以处理时间序列数据中常见的缺失值。

评估指标函数还可以接受其他额外参数，这些参数会从模型的返回值中提取并传递给指标函数。

## BasicTS 内置评估指标

BasicTS 提供了多种常用的评估指标，例如 `MAE`、`MSE`、`RMSE`、`MAPE` 和 `WAPE`。您可以在 `basicts.metrics` 模块中找到这些指标的实现。

## 如何实现自定义评估指标

根据接口规范中的指南，您可以轻松实现自定义的评估指标。以下是一个示例：

```python
class MyModel:
    def __init__(self):
        # 初始化模型
        ...
    
    def forward(...):
        # 前向计算
        ...
        return {
                'prediction': prediction,
                'targets': target,
                'other_key1': other_value1,
                'other_key2': other_value2,
                'other_key3': other_value3,
                ...
        }

def my_metric_1(prediction, targets, targets_mask=None, other_key1=None, other_key2=None, ...):
    # 计算指标
    ...

def my_metric_2(prediction, targets, targets_mask=None, other_key3=None, ...):
    # 计算指标
    ...
```

遵循这些规范，您可以灵活地在 BasicTS 中自定义和扩展评估指标，以满足特定需求。

## 🧮 仪表盘

> 该节仅涉及细节内容，绝大部分情况下不会影响使用，可以跳过。

在BasicTS中，我们使用仪表盘（`Meter`类）在训练中维护指标值。BasicTS会默认使用平均仪表盘（`AvgMeter`类），逐步更新并维护对应指标的均值，这适用于绝大部分指标。

然而，也有一些指标不应该维护均值，例如RMSE，是先求平均再开平方，此时如果逐步累积最后再求平均则会产生错误（虽然一般不影响模型的训练结果）。此时，应该使用特殊的仪表盘，实现正确的增量计算。

## 🧑‍💻 进一步探索

- **🎉 [快速上手](./getting_started_cn.md)**
- **💡 [了解 BasicTS 的设计理念](./overall_design_cn.md)**
- **📦 [探索数据集设计并自定义数据集](./dataset_design_cn.md)**
- **🛠️ [了解数据缩放器设计并创建自定义缩放器](./scaler_design_cn.md)**
- **🧠 [深入了解模型设计并构建自定义模型](./model_design_cn.md)**
- **📉 [了解评估指标设计并开发自定义损失函数与评估指标](./metrics_design_cn.md)**
- **🏃‍♂️ [掌握执行器设计并创建自定义执行器](runner_and_pipeline_cn.md)**
- **📜 [解析配置文件设计并自定义配置](./config_design_cn.md)**
- **🎯 [探索使用BasicTS进行时间序列分类](./time_series_classification_cn.md)**
- **🔍 [探索多种基线模型](../baselines/)**
