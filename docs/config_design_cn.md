# 📜 配置设计

BasicTS 的设计理念是“配置即一切“。我们的目标是让用户专注于模型和数据，而不用被繁琐的流程构建所困扰。
## 🎸 新特性
从1.0版本开始，BasicTS不再使用py文件配合命令行指定配置路径的方式，升级为使用配置类进行配置。
配置类的基类为`BasicTSConfig`，此外，每个具体任务对应一个配置类，包括`BasicTSForecastingConfig`，`BasicTSClassificationConfig`，`BasicTSImputationConfig`，`BasicTSFoundationModelConfig`等。基类`BasicTSConfig`定义了公用的字段以及保存/加载/打印配置类的方法，任务特定的配置类则包括执行该任务需要的一切配置参数。您可以灵活地向其中导入模型，并设置所有必要的选项。

配置类通常包含以下部分：

- **常规选项**: 描述一般设置，如配置说明、`gpus`、`seed` 等。
- **环境选项**: 包括设置如 `tf32`、`cudnn`、`deterministic` 等。
- **数据集选项**: 指定 **`dataset_name`（数据集名，必须显式指定）**、`dataset_type`（数据集类）、`dataset_params`（数据集参数）等。
- **数据缩放器选项**: 指定 `scaler`（缩放器类）、`norm_each_channel`（通道独立归一化）、`rescale`（是否反归一化）等。
- **模型选项**: 指定 **`model`（模型类，必须显式指定）、`model_config`（模型参数，必须显式指定）**等。
- **评估指标选项**: 包括 `metrics`（评估指标函数）、`target_metric`（目标评估指标）等。
- **训练选项**:
    - **常规**: 指定设置如 `num_epochs`/`num_steps`、`loss` 等。
    - **优化器**: 指定 `optimizer`（优化器类）、`optimizer_params`（优化器参数）等。
    - **调度器**: 指定 `lr_scheduler`（调度器类）、`lr_scheduler_params`（调度器参数）等。
    - **数据**: 指定设置如 `batch_size`、`num_workers`、`pin_memory`等。
- **验证选项**:
    - **常规**: 包括验证频率 `val_interval`。
    - **数据**: 指定设置如 `batch_size`、`num_workers`、`pin_memory` 等。
- **测试选项**:
    - **常规**: 包括测试频率 `test_interval`。
    - **数据**: 指定设置如 `batch_size`、`num_workers`、`pin_memory` 等。

Config类字段的`metadata`提供了每个字段的默认值、详细含义及使用方法。

## 🏗️ 构造配置类
### 👥 既是类，也是字典
BasicTS的配置类继承自EasyDict，既可以作为类使用，也可以作为字典使用，方便扩展并且灵活易用。下面介绍两种使用方式：
1. **像类一样使用**：一切参数都是类的字段，可以像访问类的字段一样访问、修改以及添加新的参数。例如：
	```python
	model = config.model
	config.gpus = "0"
	config.new_field = new_value
	```
2. **像字典一样使用**：一切参数都是字典的键值，可以像访问字典一样访问、修改、添加、删除参数。例如：
	```python
	model = config["model"]
	optimizer = config.get("optimizer", Adam)
	config["new_key"] = new_value
	config.pop("not_used")
	```

### 🔨 配置类中的对象
也许你已经发现了，配置中存在许多需要进一步构造的对象，例如模型、数据集、优化器、调度器等。BasicTS在配置中传入对应的类和创建对象所需的参数，而将这些对象的创建延迟到实际执行任务时。

对于这些对象，BasicTS支持两种方式来灵活地配置：
1. 将构造对象所需的全部参数以字典的形式传入。例如，将参数字典传给`dataset_params`以供后续创建数据集的实例。
	```python
	config = BasicTSForecastingConfig(
		dataset_params={
			'input_len': 336,
			'output_len': 336,
			...},
		...)
	 ```
2. 直接将构造对象需要的参数作为字段传入。例如，可以直接配置Config类的`input_len`，`output_len`，并将自定义的参数传入（不能直接在构造方法中添加未定义的字段）。
	```python
	config = BasicTSForecastingConfig(
		input_len=336,
		output_len=336,
		...)
	config.your_dataset_param_1 = param_1
	config.your_dataset_param_2 = param_2
	```

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
