# 🏃‍♂️ Runner

## 💿 Overview

The Runner is the core component of BasicTS, responsible for managing the entire training and evaluation process. It integrates various subcomponents, such as the Dataset, Scaler, Model, Metrics, and Config, to create a fair and standardized training workflow. The Runner provides several advanced features, including but not limited to:

- Early stopping
- Curriculum learning
- Gradient clipping
- Automatic model saving
- Multi-GPU training
- Persistent logging

The Runner can be used for both training and evaluating models.

## ⚡️ Training Pipeline

The typical training process using the Runner follows this structure:

```python
# Initialization
runner = Runner(config)  # Includes scaler, model, metrics, loss, optimizer, etc.

# Training
runner.train(config)
```

The `runner.train` method operates as follows:

```python
def train(config):
    init_training(config)  # Initialize training/validation/test dataloaders
    for epoch in train_epochs:
        on_epoch_start(epoch)
        for data in train_dataloader:
            loss = train_iters(data)
            optimize(loss)  # Includes backpropagation, learning rate scheduling, gradient clipping, etc.
        on_epoch_end(epoch)
    on_training_end(config)
```

### Hook Functions

The Runner provides hook functions like `on_epoch_start`, `on_epoch_end`, and `on_training_end`, allowing users to implement custom logic. For example, `on_epoch_end` can be used to evaluate validation and test sets and save intermediate models, while `on_training_end` is typically used for final evaluations and saving the final model and results.

### Training Iterations

The flow within `runner.train_iters` is structured as follows:

```python
def train_iters(data):
    data = runner.preprocessing(data)  # Normalize data
    forward_return = runner.forward(data)  # Forward pass
    forward_return = runner.postprocessing(forward_return)  # Denormalize results
    loss = runner.loss(forward_return)  # Calculate loss
    metrics = runner.metrics(forward_return)  # Calculate metrics
    return loss
```

By default, `runner.preprocessing` normalizes only the `inputs` and `target`. If additional parameters from the Dataset require normalization, you need to customize the `runner.preprocessing` function. Similarly, `runner.postprocessing` denormalizes the `inputs`, `target`, and `prediction` by default. If more parameters need denormalization, customize the `runner.postprocessing` function.

The `runner.forward` function handles data input to the model and packages the model's output into a dictionary containing `prediction`, `inputs`, `target`, and any other parameters needed for metrics calculation.

## ✨ Evaluation Pipeline

When evaluating a model using a checkpoint, the process generally follows this structure:

```python
# Initialization
runner = Runner(config)  # Includes scaler, model, metrics, loss, optimizer, etc.

# Load checkpoint
runner.load_model(checkpoint)

# Evaluation
runner.test_pipeline(config)
```

The `runner.test_pipeline` method operates as follows:

```python
def test_pipeline(config):
    init_testing(config)  # Initialize test dataloader
    all_data = []
    for data in test_dataloader:
        data = runner.preprocessing(data)  # Normalize data
        forward_return = runner.forward(data)  # Forward pass
        forward_return = runner.postprocessing(forward_return)  # Denormalize results
        all_data.append(forward_return)
    all_data = concatenate(all_data)
    metrics = runner.metrics(all_data)  # Calculate metrics
    save(forward_return, metrics)  # Optional
```

## 🛠️ Customizing the Runner

BasicTS provides a [`SimpleTimeSeriesForecastingRunner`](../basicts/runners/runner_zoo/simple_tsf_runner.py) class that handles most use cases. 

For more specific needs, you can extend the [`SimpleTimeSeriesForecastingRunner`](../basicts/runners/runner_zoo/simple_tsf_runner.py) or [`BaseTimeSeriesForecastingRunner`](../basicts/runners/base_tsf_runner.py) classes to implement functions such as `test`, `forward`, `preprocessing`, `postprocessing`, and `train_iters`.

## 🧑‍💻 Explore Further

- **🎉 [Getting Stared](./getting_started.md)**
- **💡 [Understanding the Overall Design Convention of BasicTS](./overall_design.md)**
- **📦 [Exploring the Dataset Convention and Customizing Your Own Dataset](./dataset_design.md)**
- **🛠️ [Navigating The Scaler Convention and Designing Your Own Scaler](./scaler_design.md)**
- **🧠 [Diving into the Model Convention and Creating Your Own Model](./model_design.md)**
- **📉 [Examining the Metrics Convention and Developing Your Own Loss & Metrics](./metrics_design.md)**
- **🏃‍♂️ [Mastering The Runner Convention and Building Your Own Runner](./runner_design.md)**
- **📜 [Interpreting the Config File Convention and Customizing Your Configuration](./config_design.md)**
- **🔍 [Exploring a Variety of Baseline Models](../baselines/)**
