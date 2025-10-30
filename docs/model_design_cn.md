# 🧠 模型设计

您的模型的 `forward` 函数应遵循 BasicTS 设定的规范。

## 🏗️ 构造模型

BasicTS使用配置类/字典构造模型，该配置类/字典应该包含构造模型所需的全部参数。
BasicTS模型配置类的基类为`BasicTSModelConfig`，其本身是字典的子类。当使用配置类构造模型时，您可以继承这一基类定义您的模型的配置。例如：
```python
@dataclass
class YourModelConfig(BasicTSModelConfig):
	input_len: int
	output_len: int
	num_features: int
	hidden_size: int = 256
	hidden_act: int = "relu"

class YourModel(nn.Module):
	def __init__(config: YourModelConfig):
		...
```

> [!IMPORTANT]
> ⚠️**注意**：强烈建议在配置中只使用可以JSON序列化的字段（数值、字符串、布尔、列表、元组、字典等），避免将自定义类作为字段，否则配置文件可能无法被正常保存。

## 🪴 输入接口
BasicTS 自1.0起，`forward`函数**不再强制要求传入固定的参数**（尽管未使用），而是可以按需指定传入的参数。然而，**传入参数需要遵守以下规范**。
- **标准模型参数**：BasicTS 1.0 标准的`forward`参数命名如下。模型的主输入为`inputs`，输出为`targets`；若使用时间戳，则时间戳数据为`inputs_timestamps`，`targets_timestamps`；若需要使用mask信息（如计算损失），则掩码数据为`inputs_mask`、`targets_mask`。此外，还可以传入当前训练的轮（epoch）数和步（step）数。注意，`train`参数即将被淘汰，可以访问`nn.Module`的`training`字段实现。
  ```python
  def forward(
	  self,
	  inputs: torch.Tensor,
	  targets: Optional[torch.Tensor] = None,
	  inputs_timestamps: Optional[torch.Tensor] = None,
	  targets_timestamps: Optional[torch.Tensor] = None,
	  inputs_mask: Optional[torch.Tensor] = None,
	  targets_mask: Optional[torch.Tensor] = None,
	  epoch: Optional[int] = None,
	  step: Optional[int] = None,
	  train: Optional[bool] = None
	  ,**kwargs 
  ):
  ```
  假设模型只需要用到输入序列及其时间戳，则：
  ```python
    class MyModel(nn.Module):
	    def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor):
			...
    ```
- **自定义模型参数**：您可以在`forward`函数中加入任何自定义模型参数，但需要保证数据字典中包含该键。例如：
	```python
	# 如果包含extra_flag这个额外的参数，则需要保证传入的数据字典中包含该键：
	# {"inputs": inputs, "extra_flag": extra_flag, ...}
	def forward(self, inputs: torch.Tensor, extra_flag: bool):
		...
	```
	您可以在数据流上游添加或修改数据字典：Dataset或taskflow。
	数据流向： `Dataset.__get_item__` -> `taskflow.preprocess` -> `model.forward`
	1. **在`Dataset.__get_item__`中添加（推荐）**：在数据集的`__get_item__`函数中返回包含该键的字典。
		例如：
		```python
		class MyDataset(torch.utils.data.Dataset):
			def __get_item__(self, idx: int):
				return {
					"inputs": self.inputs[idx],
					"targets": self.targets[idx],
					"extra_flag": self.flag[idx] # <-- add extra_flag
				}
		```
	2. **在`taskflow.preprocess`改变数据字典**：在自定义`Taskflow`类的`preprocess`可以修改数据字典。由于涉及对任务逻辑的修改，建议新用户谨慎使用该方法。
		例如：
		```python
		class MyTaskflow(BasicTSTaskflow):
			def preprocess(self, data: dict):
				...
				data["extra_flag"] = self.extra_flag # <-- add extra_flag
				return data
		```
## 🌷 输出接口

`forward` 函数的返回值应该是一个**字典**或一个`torch.Tensor`。
- 字典中必须包含键`prediction`，代表模型的预测结果。
- 若返回值为一个`torch.Tensor`，则后续pipeline会自动将其包装成字典`{"prediction":...}`，从而计算损失。
- 字典中可以添加任意您自定义的键，用于实现自定义逻辑或计算评估指标等。
- 想要返回在模型内部计算的损失时，必须返回包含键`loss`的字典（若直接传一个损失的`torch.Tensor`则会被视作预测结果）。当字典中包含`loss`时，后续pipeline不会再计算损失，而是直接取用。
- 想要返回在内部计算的额外损失，并与主损失相加时，须在配置类中使用`AddAuxiliaryLoss`的callback，并指定额外损失的键名。例如，传递名为`freq_loss`和`lb_loss`的额外损失，使最终损失为MSE + freq_loss + lb_loss：
 ```python
  
  # in your_train_script.py
  config=BasicTSConfig(
	  loss=masked_mse,
	  callback=[AddAuxiliaryLoss([`freq_loss`, `lb_loss`])],
	  ...
  )
  
  # in your_model.py
  def forward(...):
	  return {
		  "prediction": prediction,
		  "freq_loss": freq_loss,
		  "lb_loss": lb_loss
		  }
  ```

## 🥳 支持的基线模型

BasicTS 提供了多种内置模型。您可以在`models` 模块中找到它们，并只需导入对应的模型类和模型配置类即可使用模型。以使用STID为例：
```python
from basicts.models.STID import STID, STIDConfig

task_config = BasicTSForecastingConfig(
	model=STID,
	model_config=STIDConfig,
	...
)
```

特别地，对于内置的多任务模型，通常包含一个公用的骨干网络（`XXXBackbone`，XXX为模型名），以及若干个任务特定的模型（`XXXForYYY`，YYY为任务名）。以TimesNet为例，可以导入`TimesNetForForecasting`进行预测任务，`TimesNetForClassification`进行分类任务，`TimesNetForReconstruction`进行插补任务。这些下游任务公用相同的骨干网络和相同的配置类。
```python
from basicts.models.TimesNet import TimesNetBackbone, TimesNetForForecasting, TimesNetForClassifiction, TimesNetForReconstruction, TimesNetConfig
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
