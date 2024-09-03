# 📜 Configuration Design

The design philosophy of BasicTS is to be entirely configuration-based. Our goal is to allow users to focus on their models and data, without getting bogged down by the complexities of pipeline construction.

The configuration file is a `.py` file where you can import your model and runner, and set all necessary options. BasicTS uses EasyDict as a parameter container, making it easy to extend and flexible to use.

The configuration file typically includes the following sections:

- **General Options**: Describes general settings such as configuration description, `GPU_NUM`, `RUNNER`, etc.
- **Environment Options**: Includes settings like `TF32`, `SEED`, `CUDNN`, `DETERMINISTIC`, etc.
- **Dataset Options**: Specifies `NAME`, `TYPE` (Dataset Class), `PARAMS` (Dataset Parameters), etc.
- **Scaler Options**: Specifies `NAME`, `TYPE` (Scaler Class), `PARAMS` (Scaler Parameters), etc.
- **Model Options**: Specifies `NAME`, `TYPE` (Model Class), `PARAMS` (Model Parameters), etc.
- **Metrics Options**: Includes `FUNCS` (Metric Functions), `TARGET` (Target Metrics), `NULL_VALUE` (Handling of Missing Values), etc.
- **Train Options**:
    - **General**: Specifies settings like `EPOCHS`, `LOSS`, `EARLY_STOPPING`, etc.
    - **Optimizer**: Specifies `TYPE` (Optimizer Class), `PARAMS` (Optimizer Parameters), etc.
    - **Schduler**: Specifies `TYPE` (Scheduler Class), `PARAMS` (Scheduler Parameters), etc.
    - **Curriculum Learning**: Includes settings like `CL_EPOHS`, `WARMUP_EPOCHS`, `STEP_SIZE`, etc.
    - **Data**: Specifies settings like `BATCH_SIZE`, `NUM_WORKERS`, `PIN_MEMORY`, etc.
- **Valid Options**:
    - **General**: Includes `INTERVAL` for validation frequency.
    - **Data**: Specifies settings like `BATCH_SIZE`, `NUM_WORKERS`, `PIN_MEMORY`, etc.
- **Test Options**:
    - **General**: Includes `INTERVAL` for testing frequency.
    - **Data**: Specifies settings like `BATCH_SIZE`, `NUM_WORKERS`, `PIN_MEMORY`, etc.

For a complete guide on all configuration options and examples, refer to [examples/complete_config.py](../examples/complete_config.py).

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
