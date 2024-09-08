# 📜 配置设计 (Config)

BasicTS 的设计理念是完全基于配置。我们的目标是让用户专注于模型和数据，而不用被繁琐的流程构建所困扰。

配置文件是一个 `.py` 文件，您可以在其中导入模型和执行器，并设置所有必要的选项。BasicTS 使用 EasyDict 作为参数容器，方便扩展并且灵活易用。

配置文件通常包含以下部分：

- **常规选项**: 描述一般设置，如配置说明、`GPU_NUM`、`RUNNER` 等。
- **环境选项**: 包括设置如 `TF32`、`SEED`、`CUDNN`、`DETERMINISTIC` 等。
- **数据集选项**: 指定 `NAME`、`TYPE`（数据集类）、`PARAMS`（数据集参数）等。
- **数据缩放器选项**: 指定 `NAME`、`TYPE`（缩放器类）、`PARAMS`（缩放器参数）等。
- **模型选项**: 指定 `NAME`、`TYPE`（模型类）、`PARAMS`（模型参数）等。
- **评估指标选项**: 包括 `FUNCS`（评估指标函数）、`TARGET`（目标评估指标）、`NULL_VALUE`（缺失值处理）等。
- **训练选项**:
    - **常规**: 指定设置如 `EPOCHS`、`LOSS`、`EARLY_STOPPING` 等。
    - **优化器**: 指定 `TYPE`（优化器类）、`PARAMS`（优化器参数）等。
    - **调度器**: 指定 `TYPE`（调度器类）、`PARAMS`（调度器参数）等。
    - **课程学习**: 包括设置如 `CL_EPOCHS`、`WARMUP_EPOCHS`、`STEP_SIZE` 等。
    - **数据**: 指定设置如 `BATCH_SIZE`、`NUM_WORKERS`、`PIN_MEMORY` 等。
- **验证选项**:
    - **常规**: 包括验证频率 `INTERVAL`。
    - **数据**: 指定设置如 `BATCH_SIZE`、`NUM_WORKERS`、`PIN_MEMORY` 等。
- **测试选项**:
    - **常规**: 包括测试频率 `INTERVAL`。
    - **数据**: 指定设置如 `BATCH_SIZE`、`NUM_WORKERS`、`PIN_MEMORY` 等。

有关所有配置选项和示例的完整指南，请参阅 [examples/complete_config.py](../examples/complete_config.py)。
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
