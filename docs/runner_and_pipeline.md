# # 🏃‍♂️ Runner and Pipeline

## 💿 Overview

The runner is the core component of BasicTS, responsible for managing the entire training and evaluation process. It integrates various subcomponents such as datasets, data scalers, models, evaluation metrics, and configuration files to build a fair and extensible training and evaluation pipeline.

Starting from BasicTS 1.0, BasicTS requires only one runner class, `BasicTSRunner`, which has been completely refactored and decoupled. You no longer need to modify any runner code to implement custom extensions.

**Three-Layer Architecture of BasicTS Training and Evaluation Pipeline**: The refactored BasicTS training and evaluation pipeline can be divided into three layers:

- **Runner and General Pipeline Layer (`BasicTSRunner`)**: Contains all general processes common to the basic pipeline that are task-agnostic. Users should not directly modify code at this layer.
- **Taskflow Layer (`BasicTSTaskflow`)**: Defines task-specific steps in the basic pipeline. When not modifying the task flow, users should minimize customization of objects at this layer.
- **Callback Layer (`BasicTSCallback`)**: Defines extended functionalities beyond the basic pipeline, such as early stopping, gradient clipping, curriculum learning, etc. When extending functionality, users should implement it through callbacks whenever possible.

## ⚡️ General Pipeline

Taking training as an example (evaluation is similar), the general pipeline implemented by the runner is shown in the following pseudocode.
Consistent with standard deep learning frameworks, the **general pipeline includes: model forward pass, loss computation, loss backward pass, and optimizer update**.


```python

def train_loop(self):
	for epoch in range(num_epochs):
		
		# Event 1: on_epoch_start events
		callback_handler.trigger("on_epoch_start")
		
		for data in train_data_loder:
		
			# Event 2: on_step_start events
			callback_handler.trigger("on_step_start")
			
			# Task-specific 1: preprocess data 
			data = taskflow.preprocess(self, data)
			
			# General pipeline 1: model forward
			forward_return = forward()
			
			# Event 3: on_compute_loss events
			callback_handler.trigger("on_compute_loss")
			
			# General pipeline 2: compute loss
			loss = metric_forward(loss_function, forward_return)
			
			# Task-specific 2: get loss weight
			loss_weight = taskflow.get_weight(forward_return)
			
			# Event 4: on_backward events
			callback_handler.trigger("on_backward") # on_backward events

			# General pipeline 3: loss backward
			loss.backward()

			# Event 5: on_optimizer_step events
			callback_handler.trigger("on_optimizer_step")

			# General pipeline 4: optimizer step
			optimizer_step()

			# Task-specific 3: postprocess forward return
			forward_return = taskflow.postprocess(self, forward_return)

			# General pipeline 5: compute metrics
			metric_value = metric_forward(metric_fn, forward_return)

			# Event 6: on_step_end events
			callback_handler.trigger("on_step_end")

		# Event 7: on_epoch_end events
		callback_handler.trigger("on_epoch_end")
## ⚡️ Training Pipeline

The typical training process using the Runner follows this structure:

```python
# Initialization
runner = Runner(config)  # Includes scaler, model, metrics, loss, optimizer, etc.

# Training
runner.train(config)
```

## 💫 Taskflow

The taskflow module is located in `basicts.runners.taskflow`, and its base class is defined as follows:

```python
class BasicTSTaskflow():
	def preprocess(self, runner, data):
		pass
	
	def postprocess(self, runner, forward_return):
		pass
	
	def get_weight(self, forward_return):
		pass
```

- `preprocess`: Defines the preprocessing logic for data before the model forward pass, including normalization, generating missing value masks, etc.
- `postprocess`: Defines the postprocessing logic for data before computing metrics, including denormalization (for forecasting tasks), computing argmax (for classification tasks), etc.
- `get_weight`: Defines the loss weight of the current batch in the entire training data to ensure the overall loss of the dataset is correctly computed. For example, the weight for classification tasks should be the number of samples in the batch, while for forecasting tasks it should be the number of all valid points in the batch.

## 🪝 Callback Layer

The callback module is located in `basicts.runners.callback`. A callback class should contain several callback functions, which are called by the runner's `CallbackHandler` object at corresponding stages to enable functionality extensions.

The base callback class `BasicTSCallback` defines all available callback functions:

```python
class BasicTSCallback:
	# Called when training starts
	def on_train_start(self, runner, args, *​kwargs):
		pass
	# Called when training ends
	def on_train_end(self, runner, args, ​*​kwargs):
		pass
	# Called when an epoch starts
	def on_epoch_start(self, runner, args, ​*​kwargs):
		pass
	# Called when an epoch ends
	def on_epoch_end(self, runner, args, ​*​kwargs):
		pass
	# Called when a step starts
	def on_step_start(self, runner, args, ​*​kwargs):
		pass
	# Called when a step ends
	def on_step_end(self, runner, args, ​*​kwargs):
		pass
	# Called when validation starts
	def on_validate_start(self, runner, args, ​*​kwargs):
		pass
	# Called when validation ends
	def on_validate_end(self, runner, _args, ​_*​kwargs):
		pass
	# Called when testing starts
	def on_test_start(self, runner, _args, ​_*​kwargs):
		pass
	# Called when testing ends
	def on_test_end(self, runner, args, *​kwargs):
		pass
	# Called before computing loss
	def on_compute_loss(self, runner, _args, ​_*​kwargs):
		pass
	# Called before backward pass
	def on_backward(self, runner, _args, ​_*​kwargs):
		pass
	# Called before optimizer update
	def on_optimizer_step(self, runner, _args, ​_*​kwargs):
		pass
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
