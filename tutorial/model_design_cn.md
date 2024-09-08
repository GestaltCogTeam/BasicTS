# 🧠 模型设计 (Model)

您的模型的 `forward` 函数应遵循 BasicTS 设定的规范。

## 🪴 输入接口

BasicTS 会将以下参数传递给模型的 `forward` 函数：

- **history_data** (`torch.Tensor`): 历史数据，形状为 `[B, L, N, C]`，其中 `B` 代表批次大小，`L` 为序列长度，`N` 为节点数量，`C` 为特征数量。
- **future_data** (`torch.Tensor` 或 `None`): 未来数据，形状为 `[B, L, N, C]`。如果未来数据不可用（例如在测试阶段），则此参数为 `None`。
- **batch_seen** (`int`): 目前处理的批次数。
- **epoch** (`int`): 当前的训练轮数。
- **train** (`bool`): 表示模型是否处于训练模式。

## 🌷 输出接口

`forward` 函数的输出可以是一个形状为 `[B, L, N, C]` 的 `torch.Tensor`，其中通常 `C=1`，表示预测的值。

或者，模型可以返回一个包含 `prediction` 键的字典，其中 `prediction` 包含上述描述的预测值。该字典还可以包含其他自定义键，作为损失函数和评估指标的参数。更多细节可以在 [评估指标](./metrics_design_cn.md) 中找到。

一个示例可以在 [多层感知机（MLP）模型](../examples/arch.py) 中找到。


## 🥳 支持的基线模型

BasicTS 提供了多种内置模型。您可以在 [baselines](../baselines/) 文件夹中找到它们。要运行一个基线模型，可以使用以下命令：

```bash
python experiments/train.py -c baselines/${MODEL_NAME}/${DATASET_NAME}.py -g '{GPU_IDs}'
```

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
