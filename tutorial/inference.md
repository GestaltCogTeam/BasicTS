# Inference (Experimental)

This tutorial introduces how to inference in BasicTS.

## 🗒 Inference Script

Using the inference script, you can read input data from a specified file, perform inference with a specified model, and save the output results to a file.

### Model Preparation

Before using the inference feature, you need to train a model first. Model training can be referenced in the [Quick Start](https://github.com/GestaltCogTeam/BasicTS/blob/master/tutorial/getting_started.md) section.

### Data Preparation

Input data should be in CSV format, UTF-8 encoded, and comma-separated.  
The first column is the timestamp in the format `Year-Month-Day Hour:Minute:Second`.  
Subsequent columns are data columns.

Example input data:

```csv
2024-01-01 00:00:00,1,2,3,4,5,6,7
2024-01-01 00:05:00,1,2,3,4,5,6,7
2024-01-01 00:10:00,1,2,3,4,5,6,7
2024-01-01 00:15:00,1,2,3,4,5,6,7
2024-01-01 00:20:00,1,2,3,4,5,6,7
2024-01-01 00:25:00,1,2,3,4,5,6,7
2024-01-01 00:30:00,1,2,3,4,5,6,7
```

### Using the Inference Script

The following parameters are required for inference:

- Model checkpoint path
- Configuration file path
- Input data path
- Output data path
- Available GPU list (optional)

For UTFS models (e.g., TimeMoe), additional parameters are required:

- Context length
- Prediction length

The inference script is located at `experiments/inference.py`. 

Start it with the following command:

```bash
python experiments/inference.py -cfg <config_path> -ckpt <checkpoint_path> -i <input_path> -o <output_path>
``` 

For UTFS models, include context and prediction lengths:

```bash
python experiments/inference.py -cfg <config_path> -ckpt <checkpoint_path> -i <input_path> -o <output_path> -ctx <context_length> -pred <prediction_length>
``` 

### Examples

1.Using STID on ETTh1 dataset:

```bash
python experiments/inference.py -cfg "baselines/STID/ETTh1.py" -ckpt "/checkpoints/STID/ETTh1_100_336_336/587c21xxxx/STID_best_val_MAE.pt" -i "./in_etth1.csv" -o "out.csv"
```

Input data:

```csv
2024-01-01 00:00:00,1,2,3,4,5,6,7
2024-01-01 00:05:00,1,2,3,4,5,6,7
2024-01-01 00:10:00,1,2,3,4,5,6,7
2024-01-01 00:15:00,1,2,3,4,5,6,7
2024-01-01 00:20:00,1,2,3,4,5,6,7
2024-01-01 00:25:00,1,2,3,4,5,6,7
...
```

Results:

```csv
2024-01-03 13:00:00,-1.1436124,-0.0042671096,-0.35258546,1.7036028,2.1393495,8.280911,-1.0798432
2024-01-03 14:00:00,-1.1344103,-0.0021482962,-0.3535639,1.71777,2.1475496,8.356691,-1.0723352
2024-01-03 15:00:00,-1.1335654,0.004845081,-0.34263217,1.7284462,2.1487343,8.307554,-1.0726832
2024-01-03 16:00:00,-1.136602,-0.0066127134,-0.35953742,1.7203176,2.1453576,8.329743,-1.0784494
2024-01-03 17:00:00,-1.139412,0.0039077974,-0.35144827,1.7184068,2.1413972,8.300636,-1.074183
...
```

1. Using Chronos:

```bash
python experiments/inference.py -cfg "baselines/ChronosBolt/config/chronos_base.py" -ckpt "ckpts_release/ChronosBolt-base-BLAST.pt" -i "./in_etth1.csv" -o "out.csv" -ctx 72 -pred 36
```

Input data:

```csv
2024-01-01 00:00:00,1,2,3,4,5,6,7
2024-01-01 00:05:00,1,2,3,4,5,6,7
2024-01-01 00:10:00,1,2,3,4,5,6,7
2024-01-01 00:15:00,1,2,3,4,5,6,7
2024-01-01 00:20:00,1,2,3,4,5,6,7
2024-01-01 00:25:00,1,2,3,4,5,6,7
...
```

Results:

```csv
2024-01-03 12:05:00,1.0,2.0,3.0,4.0,5.0,6.0,7.0
2024-01-03 12:10:00,1.0,2.0,3.0,4.0,5.0,6.0,7.0
2024-01-03 12:15:00,1.0,2.0,3.0,4.0,5.0,6.0,7.0
2024-01-03 12:20:00,1.0,2.0,3.0,4.0,5.0,6.0,7.0
2024-01-03 12:25:00,1.0,2.0,3.0,4.0,5.0,6.0,7.0
...
```


## 🌐 Web Page (Coming Soon)

A visual inference interface via web page.

## 🖥 API Service (Coming Soon)

HTTP-based inference API service.
> [!NOTE]  
> The API service is not optimized for high-concurrency scenarios. Do not use it directly in the production environment.


## 🧑‍💻 Explore Further

This tutorial has equipped you with the fundamentals to get started with BasicTS, but there’s much more to discover. Before delving into advanced topics, let’s take a closer look at the structure of BasicTS:

<div align="center">
  <img src="figures/DesignConvention.jpeg" height=350>
</div>

The core components of BasicTS include `Dataset`, `Scaler`, `Model`, `Metrics`, `Runner`, and `Config`. To streamline the debugging process, BasicTS operates as a localized framework, meaning all the code runs directly on your machine. There’s no need to pip install basicts; simply clone the repository, and you’re ready to run the code locally.

Below are some advanced topics and additional features to help you maximize the potential of BasicTS:

- **🎉 [Getting Stared](./getting_started.md)**
- **💡 [Understanding the Overall Design Convention of BasicTS](./overall_design.md)**
- **📦 [Exploring the Dataset Convention and Customizing Your Own Dataset](./dataset_design.md)**
- **🛠️ [Navigating The Scaler Convention and Designing Your Own Scaler](./scaler_design.md)**
- **🧠 [Diving into the Model Convention and Creating Your Own Model](./model_design.md)**
- **📉 [Examining the Metrics Convention and Developing Your Own Loss & Metrics](./metrics_design.md)**
- **🏃‍♂️ [Mastering The Runner Convention and Building Your Own Runner](./runner_design.md)**
- **📜 [Interpreting the Config File Convention and Customizing Your Configuration](./config_design.md)**
- **🔍 [Exploring a Variety of Baseline Models](../baselines/)**
