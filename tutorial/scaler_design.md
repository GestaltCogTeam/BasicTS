# 🛠️ Scaler Design

## 🧐 What is a Scaler and Why Do We Need It?

A scaler is a class designed to handle the normalization and denormalization of data. In time series analysis, raw data often exhibits significant variations in scale. As a result, models—especially deep learning models—typically do not operate directly on the raw data. Instead, a scaler is used to normalize the data within a specific range, making it more suitable for modeling. When calculating loss functions or evaluation metrics, the data may be denormalized back to its original scale to ensure accurate comparison.

This makes the scaler an essential component in the time series analysis workflow.

## 👾 How is the Scaler Initialized and When Does It Function?

The scaler is initialized alongside other components within the runner.

For example, a Z-Score scaler reads the raw data and computes the mean and standard deviation based on the training data.

The scaler functions after the data is extracted from the dataset. The data is first normalized by the scaler before being passed to the model for training. After the model processes the data, the scaler denormalizes the output before it is passed to the runner for loss calculation and metric evaluation.

> [!IMPORTANT]  
> In many time series analysis studies, normalization often occurs during data preprocessing, as was the case in earlier versions of BasicTS. However, this approach is not scalable. Adjustments like changing input/output lengths, applying different normalization ways (e.g., individual normalization for each time series), or altering the training/validation/test split ratios would require re-preprocessing the data. To overcome this limitation, BasicTS adopts an "instant normalization" approach, where data is normalized each time it is extracted.

```python
# in runner
for data in dataloader:
    data = scaler.transform(data)
    forward_return = forward(data)
    forward_return = scaler.inverse_transform(forward_return)
```

## 🧑‍🔧 How to Select or Customize a Scaler

BasicTS provides several common scalers, such as the Z-Score scaler and Min-Max scaler. You can easily switch scalers by setting `CFG.SCALER.TYPE` in the configuration file.

If you need to customize a scaler, you can extend the `basicts.scaler.BaseScaler` class and implement the `transform` and `inverse_transform` methods. Alternatively, you can choose not to extend the class but must still implement these two methods.

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
