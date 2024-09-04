# 🧠 Model Design

The `forward` function of your model should adhere to the conventions established by BasicTS.

## 🪴 Input Interface

BasicTS will pass the following arguments to the `forward` function of your model:

- **history_data** (`torch.Tensor`): Historical data with shape `[B, L, N, C]`, where `B` represents the batch size, `L` is the sequence length, `N` is the number of nodes, and `C` is the number of features.
- **future_data** (`torch.Tensor` or `None`): Future data with shape `[B, L, N, C]`. This can be `None` if future data is not available (e.g., during the testing phase).
- **batch_seen** (`int`): The number of batches processed so far.
- **epoch** (`int`): The current epoch number.
- **train** (`bool`): Indicates whether the model is in training mode.

## 🌷 Output Interface

The output of the `forward` function can be a `torch.Tensor` representing the predicted values with shape `[B, L, N, C]`, where typically `C=1`.

Alternatively, the model can return a dictionary that must include the key `prediction`, which contains the predicted values as described above. This dictionary can also include additional custom keys that correspond to arguments for the loss and metrics functions.

An example can be found in the [Multi-Layer Perceptron (MLP) model](../examples/arch.py).

## 🥳 Supported Baslines

BasicTS provides a variety of built-in models. You can find them in [baselines](../baselines/) folder. To run a baseline model, use the following command:

```bash
python experiments/train.py -c baselines/${MODEL_NAME}/${DATASET_NAME}.py -g '{GPU_IDs}'
```

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
