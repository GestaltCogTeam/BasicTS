# 📉 Metrics Design

## Interface Conventions

Metrics are essential components for evaluating model performance. In BasicTS, metrics are functions that take the model's predictions, ground truth values, and other parameters as inputs and return a scalar value to assess the model's performance.

A well-defined metric function should include the following parameters:
- **prediction**: The predicted values from the model
- **target**: The actual ground truth values
- **null_val**: An optional parameter to handle missing values

The `prediction` and `target` parameters are mandatory, while the `null_val` parameter is optional but strongly recommended for handling missing values, which are common in time series data. 
The `null_val` is automatically set based on the `CFG.NULL_VAL` value in the configuration file, which defaults to `np.nan`.

Metric functions can also accept additional parameters, which are extracted from the model's return values and passed to the metric function. 

> [!CAUTION]  
> If these additional parameters (besides `prediction`, `target`, and `inputs`) require normalization or denormalization, please adjust the `preprocessing` and `postprocessing` functions in the `runner` accordingly.

## Built-in Metrics in BasicTS

BasicTS comes with several commonly used metrics, such as `MAE`, `MSE`, `RMSE`, `MAPE`, and `WAPE`. You can find these metrics implemented in the `basicts/metrics` directory.

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
                'target': target,
                'other_key1': other_value1,
                'other_key2': other_value2,
                'other_key3': other_value3,
                ...
        }

def my_metric_1(prediction, target, null_val=np.nan, other_key1=None, other_key2=None, ...):
    # Calculate the metric
    ...

def my_metric_2(prediction, target, null_val=np.nan, other_key3=None, ...):
    # Calculate the metric
    ...
```

By adhering to these conventions, you can flexibly customize and extend the metrics in BasicTS to meet specific requirements.

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
