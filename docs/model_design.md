# 🧠 Model Design

Your model's `forward` function should adhere to the specifications set by BasicTS.

## 🏗️ Constructing Models

BasicTS uses configuration classes/dictionaries to construct models, which should contain all parameters required for model construction.
The base configuration class for BasicTS models is `BasicTSModelConfig`, which is a subclass of `dict`. When using configuration classes to construct models, you can inherit from this base class to define your model's configuration. For example:
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
> ⚠️ **Note**: It is strongly recommended to use only JSON-serializable fields (numbers, strings, booleans, lists, tuples, dictionaries, etc.) in configurations and avoid using custom classes as fields, otherwise configuration files may not be saved properly.

## 🪴 Input Interface

Starting from BasicTS 1.0, the `forward` function **no longer requires fixed parameters** (even if unused), but can specify parameters as needed. However, **input parameters must adhere to the following specifications**.

- **Standard Model Parameters**: The standard `forward` parameter names in BasicTS 1.0 are as follows. The model's main input is `inputs`, output is `targets`; if using timestamps, timestamp data is `inputs_timestamps`, `targets_timestamps`; if mask information is needed (e.g., for loss calculation), mask data is `inputs_mask`, `targets_mask`. Additionally, the current training epoch and step number can be passed. Note that the `train` parameter is being phased out and can be replaced by accessing the `training` field of `nn.Module`.
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
  Assuming the model only needs the input sequence and its timestamps:
  ```python
    class MyModel(nn.Module):
	    def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor):
			...
    ```
- **Custom Model Parameters**: You can add any custom parameters to the `forward` function, but you must ensure the data dictionary contains this key. For example:
	```python
	# If including an extra parameter like extra_flag, ensure the input data dictionary contains this key:
	# {"inputs": inputs, "extra_flag": extra_flag, ...}
	def forward(self, inputs: torch.Tensor, extra_flag: bool):
		...
	```
	You can add or modify the data dictionary upstream in the data flow: in the Dataset or taskflow.
	Data flow: `Dataset.__get_item__` -> `taskflow.preprocess` -> `model.forward`
	1. **Add in `Dataset.__get_item__` (Recommended)**: Return a dictionary containing the key in the dataset's `__get_item__` function. For example:
		```python
		class MyDataset(torch.utils.data.Dataset):
			def __get_item__(self, idx: int):
				return {
					"inputs": self.inputs[idx],
					"targets": self.targets[idx],
					"extra_flag": self.flag[idx] # <-- add extra_flag
				}
		```
	2. **Modify data dictionary in `taskflow.preprocess`**: The `preprocess` method of a custom `Taskflow` class can modify the data dictionary. Since this involves modifying task logic, new users are advised to use this method with caution. For example:
		```python
		class MyTaskflow(BasicTSTaskflow):
			def preprocess(self, data: dict):
				...
				data["extra_flag"] = self.extra_flag # <-- add extra_flag
				return data
		```
## 🌷 Output Interface

The return value of the `forward` function should be a **dictionary** or a `torch.Tensor`.

- The dictionary must contain the key `prediction`, representing the model's prediction results.
- If the return value is a `torch.Tensor`, the subsequent pipeline will automatically wrap it into a dictionary `{"prediction":...}` for loss calculation.
- The dictionary can include any custom keys you define for implementing custom logic or calculating evaluation metrics, etc.
- To return a loss calculated internally by the model, you must return a dictionary containing the key `loss` (if you directly return a loss `torch.Tensor`, it will be treated as a prediction result). When the dictionary contains `loss`, the subsequent pipeline will not calculate the loss but will use it directly.
- To return additional losses calculated internally and sum them with the main loss, you must use the `AddAuxiliaryLoss` callback in the configuration class and specify the key names of the additional losses. For example, to pass additional losses named `freq_loss` and `lb_loss`, making the final loss MSE + freq_loss + lb_loss:
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

## 🥳 Supported Baseline Models

BasicTS provides various built-in models. You can find them in the `models` module and simply import the corresponding model class and model configuration class to use the model. For example, using STID:
```python
from basicts.models.STID import STID, STIDConfig

task_config = BasicTSForecastingConfig(
	model=STID,
	model_config=STIDConfig,
	...
)
```

Specifically, for built-in multi-task models, they typically include a shared backbone network (`XXXBackbone`, where XXX is the model name) and several task-specific models (`XXXForYYY`, where YYY is the task name). For example, with TimesNet, you can import `TimesNetForForecasting` for forecasting tasks, `TimesNetForClassification` for classification tasks, and `TimesNetForReconstruction` for imputation tasks. These downstream tasks share the same backbone network and configuration class.
```python
from basicts.models.TimesNet import TimesNetBackbone, TimesNetForForecasting, TimesNetForClassifiction, TimesNetForReconstruction, TimesNetConfig
```
## 🧑‍💻 Explore Further

- **🎉 [Getting Stared](./getting_started.md)**
- **💡 [Understanding the Overall Design Convention of BasicTS](./overall_design.md)**
- **📦 [Exploring the Dataset Convention and Customizing Your Own Dataset](./dataset_design.md)**
- **🛠️ [Navigating The Scaler Convention and Designing Your Own Scaler](./scaler_design.md)**
- **🧠 [Diving into the Model Convention and Creating Your Own Model](./model_design.md)**
- **📉 [Examining the Metrics Convention and Developing Your Own Loss & Metrics](./metrics_design.md)**
- **🏃‍♂️ [Mastering The Runner Convention and Building Your Own Runner](runner_and_pipeline.md)**
- **📜 [Interpreting the Config File Convention and Customizing Your Configuration](./config_design.md)**
- **🎯 [Exploring Time Series Classification with BasicTS](./time_series_classification_cn.md)**
- **🔍 [Exploring a Variety of Baseline Models](../baselines/)**
