# 📜 Configuration Design

The design philosophy of **BasicTS** is *"All-in-configuration"*.  
Our goal is to let users focus on models and data, without being burdened by complex training pipelines.

## 🎸 New Features
Starting from version 1.0, **BasicTS** no longer uses `.py` files combined with command-line configuration paths.  
Instead, it has been upgraded to a **configuration class–based system**.

The base configuration class is `BasicTSConfig`.  
In addition, each specific task has its own configuration class, including:
- `BasicTSForecastingConfig`
- `BasicTSClassificationConfig`
- `BasicTSImputationConfig`
- `BasicTSFoundationModelConfig`

The base class `BasicTSConfig` defines shared fields and methods for saving, loading, and printing configurations,  
while task-specific configuration classes contain all parameters required for executing that task.  
Users can flexibly import models and set all necessary options within these configuration classes.

A configuration class typically includes the following sections:

- **General Options**: Basic settings such as configuration description, `gpus`, and `seed`.
- **Environment Options**: Settings such as `tf32`, `cudnn`, and `deterministic`.
- **Dataset Options**: Specify **`dataset_name` (dataset name, must be explicitly defined)**,  
  `dataset_type` (dataset class), and `dataset_params` (dataset parameters).
- **Scaler Options**: Specify `scaler` (scaler class), `norm_each_channel` (per-channel normalization),  
  and `rescale` (whether to inverse-normalize outputs).
- **Model Options**: Specify **`model` (model class, must be explicitly defined)** and  
  **`model_config` (model parameters, must be explicitly defined)**.
- **Evaluation Metrics Options**: Define `metrics` (evaluation functions) and `target_metric` (primary metric).
- **Training Options**:
    - **General**: Define settings such as `num_epochs` / `num_steps`, and `loss`.
    - **Optimizer**: Specify `optimizer` (optimizer class) and `optimizer_params` (optimizer parameters).
    - **Scheduler**: Specify `lr_scheduler` (scheduler class) and `lr_scheduler_params` (scheduler parameters).
    - **Data**: Define `batch_size`, `num_workers`, and `pin_memory`.
- **Validation Options**:
    - **General**: Validation frequency `val_interval`.
    - **Data**: Define `batch_size`, `num_workers`, and `pin_memory`.
- **Testing Options**:
    - **General**: Testing frequency `test_interval`.
    - **Data**: Define `batch_size`, `num_workers`, and `pin_memory`.

Each field in the Config class contains `metadata` that provides default values, detailed explanations, and usage instructions.

## 🏗️ Building Configuration Classes
### 👥 Both a Class and a Dictionary
The configuration classes in **BasicTS** inherit from **EasyDict**,  
so they can be used both as **Python classes** and **dictionaries**,  
making them highly flexible and extensible.  

There are two main ways to use them:

1. **As a class** — all parameters are accessible as class attributes.  
   You can access, modify, or add new fields just like regular class variables:
    ```python
    model = config.model
    config.gpus = "0"
    config.new_field = new_value
    ```

2. **As a dictionary** — all parameters are accessible as key-value pairs.  
   You can access, modify, add, or delete parameters just like with a dictionary:
    ```python
    model = config["model"]
    optimizer = config.get("optimizer", Adam)
    config["new_key"] = new_value
    config.pop("not_used")
    ```

### 🔨 Objects Inside Configuration Classes
You may have noticed that many configuration fields represent objects that need to be constructed later,  
such as models, datasets, optimizers, and schedulers.  
In BasicTS, you specify the corresponding class and initialization parameters in the configuration,  
and the actual object construction is deferred until runtime.

There are two flexible ways to define these objects:

1. **Pass all initialization parameters as a dictionary.**  
   For example, use `dataset_params` to specify parameters for dataset creation:
    ```python
    config = BasicTSForecastingConfig(
        dataset_params={
            'input_len': 336,
            'output_len': 336,
            ...
        },
        ...
    )
    ```

2. **Pass parameters directly as configuration fields.**  
   For example, you can directly set `input_len` and `output_len`,  
   and add custom fields later (note: only predefined fields can be passed during initialization):
    ```python
    config = BasicTSForecastingConfig(
        input_len=336,
        output_len=336,
        ...
    )
    config.your_dataset_param_1 = param_1
    config.your_dataset_param_2 = param_2
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
