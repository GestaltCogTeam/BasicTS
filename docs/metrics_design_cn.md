# 📉 评估指标设计 (Metrics)

## 接口规范

评估指标是评估模型性能的重要组成部分。在 BasicTS 中，评估指标是接受模型预测值、真实值及其他参数作为输入并返回标量值以评估模型性能的函数。

一个定义良好的评估指标函数应包含以下参数：
- **prediction**: 模型的预测值
- **target**: 实际的真实值
- **null_val**: 可选参数，用于处理缺失值

`prediction` 和 `target` 是必需参数，而 `null_val` 是可选参数，但强烈建议采纳，以处理时间序列数据中常见的缺失值。`null_val` 会自动基于配置文件中的 `CFG.NULL_VAL` 值设置，缺省值为 `np.nan`。

评估指标函数还可以接受其他额外参数，这些参数会从模型的返回值中提取并传递给指标函数。

> [!CAUTION]  
> 如果这些额外参数（如 `prediction`、`target` 和 `inputs` 之外的参数）需要归一化或反归一化，请相应地调整 `runner` 中的 `preprocessing` 和 `postprocessing` 函数。

## BasicTS 内置评估指标

BasicTS 提供了多种常用的评估指标，例如 `MAE`、`MSE`、`RMSE`、`MAPE` 和 `WAPE`。您可以在 `basicts/metrics` 目录中找到这些指标的实现。

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
                'target': target,
                'other_key1': other_value1,
                'other_key2': other_value2,
                'other_key3': other_value3,
                ...
        }

def my_metric_1(prediction, target, null_val=np.nan, other_key1=None, other_key2=None, ...):
    # 计算指标
    ...

def my_metric_2(prediction, target, null_val=np.nan, other_key3=None, ...):
    # 计算指标
    ...
```

遵循这些规范，您可以灵活地在 BasicTS 中自定义和扩展评估指标，以满足特定需求。

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
