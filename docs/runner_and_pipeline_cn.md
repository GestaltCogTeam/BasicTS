# 🏃‍♂️ 执行器与流程

## 💿 概述

执行器是 BasicTS 的核心组件，负责管理整个训练和评估过程。它将数据集、数据缩放器、模型、评估指标和配置文件等各个子组件集成在一起，构建一个公平且可扩展的训练和评估流程。

自BasicTS 1.0起，BasicTS只需要一个执行器类`BasicTSRunner`，并对其进行了全面重构和解耦。您无需再修改任何执行器代码，就能实现任何自定义的扩展功能。

**BasicTS训练与评估流程的三层架构**：重构后的BasicTS的训练与评估流程可以被分为三个层次。

- **执行器与通用流程层（`BasicTSRunner`）**：集结了一切基础流程中通用的、和具体任务无关的训练流程。用户不应该直接修改该层次的代码。
- **任务流层（`BasicTSTaskflow`）**：定义了基础流程中和任务相关的步骤。当不修改任务流程时，用户应该尽量少地自定义该层的对象。
- **回调层（`BasicTSCallback`）**：定义了基础流程之外的扩展功能，例如早停、梯度裁剪、课程学习等。当想要扩展功能时，用户应该尽可能地通过回调来实现。

## ⚡️ 通用流程

以训练为例（评估类似），执行器实现的通用流程如下列伪代码所示。
与标准深度学习框架相符，**通用流程包括：模型前传、计算损失、损失反传、优化器更新**。

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
```

## 💫 任务流

任务流模块位于`basicts.runners.taskflow`，其基类定义如下：

```python
class BasicTSTaskflow():
	def preprocess(self, runner, data):
		pass
	
	def postprocess(self, runner, forward_return):
		pass
	
	def get_weight(self, forward_return):
		pass
```

- `preprocess`：定义数据在模型前传前的预处理逻辑，包括归一化、生成缺失值掩码等。
- `postprocess`：定义数据在计算指标前的后处理逻辑，包括反归一化（预测任务），计算argmax（分类任务）等。
- `get_weight`：定义当前批次在全部训练数据中的损失权重，保证数据集的整体损失能被正确计算。例如，分类任务的权重应该是该批次的样本数，预测任务应该是该批次全部有效点的数量。

## 🪝 回调层

回调模块位于`basicts.runners.callback`。一个回调类应该包含若干个回调函数，执行器的`CallbackHandler`对象会在对应的阶段调用这些函数，以实现功能的扩展。

回调基类`BasicTSCallback`定义了全部可用的回调函数：
```python
class BasicTSCallback:
	# 训练开始时
	def on_train_start(self, runner, *args, **kwargs):
		pass
	# 训练结束时
	def on_train_end(self, runner, *args, **kwargs):
		pass
	# epoch开始时
	def on_epoch_start(self, runner, *args, **kwargs):
		pass
	# epoch结束时
	def on_epoch_end(self, runner, *args, **kwargs):
		pass
	# step开始时
	def on_step_start(self, runner, *args, **kwargs):
		pass
	# step结束时
	def on_step_end(self, runner, *args, **kwargs):
		pass
	# 验证开始时
	def on_validate_start(self, runner, *args, **kwargs):
		pass
	# 验证结束时
	def on_validate_end(self, runner, *args, **kwargs):
		pass
	# 测试开始时
	def on_test_start(self, runner, *args, **kwargs):
		pass
	# 测试结束时
	def on_test_end(self, runner, *args, **kwargs):
		pass
	# 计算损失前
	def on_compute_loss(self, runner, *args, **kwargs):
		pass
	# 反向传播前
	def on_backward(self, runner, *args, **kwargs):
		pass
	# 优化器更新前
	def on_optimizer_step(self, runner, *args, **kwargs):
		pass
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
