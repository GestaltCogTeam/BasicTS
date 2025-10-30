# 📉 Metrics Design

## Interface Specification

Metrics are an essential component for evaluating model performance. In BasicTS, a metric is a function that takes model predictions, ground truth values, and other parameters as input and returns a scalar value to assess model performance.

A well-defined metric function should include the following parameters:
- **prediction**: The model's predictions
- **targets**: The actual ground truth values
- **targets_mask**: An optional parameter used to specify which points to compute the loss on (commonly used to mask missing values)

`prediction` and `targets` are required parameters, while `targets_mask` is optional but highly recommended to handle missing values commonly found in time series data.

Metric functions can also accept other additional parameters, which will be extracted from the model's return value and passed to the metric function.

## Built-in Metrics in BasicTS

BasicTS provides a variety of commonly used metrics, such as `MAE`, `MSE`, `RMSE`, `MAPE`, and `WAPE`. You can find the implementations of these metrics in the `basicts.metrics` module.

## How to Implement Custom Metrics

Following the guidelines outlined in the Interface Conventions section, you can easily implement custom metrics. Here’s an example:

```python
class MyModel:
    def __init__(self):
        # Initialize the model
        ...
    
    def forward(...):
        # Forward computation
        ...
        return {
                'prediction': prediction,
                'targets': target,
                'other_key1': other_value1,
                'other_key2': other_value2,
                'other_key3': other_value3,
                ...
        }

def my_metric_1(prediction, targets, targets_mask=None, other_key1=None, other_key2=None, ...):
    # Calculate the metric
    ...

def my_metric_2(prediction, targets, targets_mask=None, other_key3=None, ...):
    # Calculate the metric
    ...
```

By adhering to these conventions, you can flexibly customize and extend the metrics in BasicTS to meet specific requirements.

## 🧮 Meter

> This section covers implementation details. It will not affect usage in most cases and can be skipped.

In BasicTS, we use a `Meter` class to maintain metric values during training. BasicTS uses `AvgMeter` class by default, which incrementally updates and maintains the average value of the corresponding metric. This is suitable for most metrics.

However, some metrics should not maintain a simple average. For example, RMSE involves taking the square root after averaging; incrementally updating and averaging at the end can lead to incorrect results (although it generally does not affect the final model training outcome). In such cases, a specialized meter should be used to implement the correct incremental calculation.

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
