############################## 导入依赖 ##############################
import os

from easydict import EasyDict

# 导入数据集类
from basicts.data import TimeSeriesForecastingDataset
# 导入指标和损失函数
from basicts.metrics import masked_mae, masked_mape, masked_rmse
# 导入执行器类
from basicts.runners import SimpleTimeSeriesForecastingRunner
# 导入缩放器类
from basicts.scaler import ZScoreScaler
# 导入数据集配置
from basicts.utils import get_regular_settings

# 导入模型架构
from .arch import MultiLayerPerceptron as MLP

############################## 热门参数 ##############################

# 数据集和指标配置
DATA_NAME = 'PEMS08'  # 数据集名称
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']  # 输入序列长度
OUTPUT_LEN = regular_settings['OUTPUT_LEN']  # 输出序列长度
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']  # 训练/验证/测试集比例
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL'] # 是否对数据的每个通道独立归一化（例如，独立计算均值和标准差）
RESCALE = regular_settings['RESCALE'] # 是否对数据进行重新缩放
NULL_VAL = regular_settings['NULL_VAL'] # 数据中的空值

# 模型架构和参数
MODEL_ARCH = MLP
MODEL_PARAM = {
    'history_seq_len': INPUT_LEN,
    'prediction_seq_len': OUTPUT_LEN,
    'hidden_dim': 64
}
NUM_EPOCHS = 100

############################## 通用配置 ##############################

CFG = EasyDict()

# 通用设置
CFG.DESCRIPTION = '一个示例配置' # 配置的描述，不用于 BasicTS 中，但对用户记住配置的用途有帮助
CFG.GPU_NUM = 1 # 使用的 GPU 数量（0 表示使用 CPU 模式）

# 执行器
CFG.RUNNER = SimpleTimeSeriesForecastingRunner # 执行器类

############################## 环境配置 ##############################

CFG.ENV = EasyDict() # 环境设置。默认值：None

# GPU 和随机种子设置
CFG.ENV.TF32 = False # 是否在 GPU 中使用 TensorFloat-32。默认值：False。详见 https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere
CFG.ENV.SEED = 42 # 随机种子。默认None
CFG.ENV.DETERMINISTIC = False # 是否设置随机种子以获得确定性的结果。默认值：False
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True # 是否启用 cuDNN。默认值：True
CFG.ENV.CUDNN.BENCHMARK = True # 是否启用 cuDNN 基准测试。默认值：True
CFG.ENV.CUDNN.DETERMINISTIC = False # 是否将 cuDNN 设置为确定性模式。默认值：False

############################## 数据集配置 ##############################

CFG.DATASET = EasyDict() # 数据集设置。默认值：None。如果未指定，从 CFG.[TRAIN, VAL, TEST].DATA.DATASET 获取训练、验证和测试数据集。

# 数据集设置
CFG.DATASET.NAME = DATA_NAME # 数据集名称，用于保存检查点和设置进程标题。
CFG.DATASET.TYPE = TimeSeriesForecastingDataset # 训练、验证和测试中使用的数据集类。
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
    # 'mode' 由执行器自动设置
}) # 数据集类的参数

############################## 缩放器配置 ##############################

CFG.SCALER = EasyDict() # 缩放器设置。默认值：None。如果未指定，数据将直接用于训练、验证和测试。

# 缩放器设置
CFG.SCALER.TYPE = ZScoreScaler # 缩放器类
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
}) # 缩放器类的参数

############################## 模型配置 ##############################

CFG.MODEL = EasyDict() # 模型设置，必须指定。

# 模型设置
CFG.MODEL.NAME = MODEL_ARCH.__name__ # 模型名称，必须指定，用于保存检查点和设置进程标题。
CFG.MODEL.ARCH = MODEL_ARCH # 模型架构，必须指定。
CFG.MODEL.PARAM = MODEL_PARAM # 模型参数，必须指定。
# 作为输入使用的特征。输入数据的大小通常为 [B, L, N, C]，
# 此参数指定最后一个维度的索引，即 history_data[:, :, :, CFG.MODEL.FORWARD_FEATURES]。
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
# 作为输出使用的特征。目标数据的大小通常为 [B, L, N, C]，此参数指定最后一个维度的索引，即 future_data[:, :, :, CFG.MODEL.TARGET_FEATURES]。
CFG.MODEL.TARGET_FEATURES = [0]
# 待预测的时间序列索引，默认为None。该参数在多变量到单变量预测（Multivariate-to-Univariate）的场景下特别有用。
# 例如，当输入7条时间序列时，若需要预测最后两条序列，可以通过设置`CFG.MODEL.TARGET_TIME_SERIES=[5, 6]`来实现。
CFG.MODEL.TARGET_TIME_SERIES = None
# 是否设置计算图。默认值：False。许多论文的实现（如 DCRNN，GTS）类似于 TensorFlow，需要第一次前向传播时建立计算图并创建参数。
CFG.MODEL.SETUP_GRAPH = False
# 控制 torch.nn.parallel.DistributedDataParallel 的 `find_unused_parameters` 参数。
# 在分布式计算中，如果前向传播过程中存在未使用的参数，PyTorch 通常会抛出 RuntimeError。在这种情况下，应将此参数设置为 True。
CFG.MODEL.DDP_FIND_UNUSED_PARAMETERS = False

############################## 指标配置 ##############################

CFG.METRICS = EasyDict() # 指标设置。默认值：None。如果未指定，将使用默认指标。

# 指标设置
CFG.METRICS.FUNCS = EasyDict({
                                'MAE': masked_mae,
                                'MAPE': masked_mape,
                                'RMSE': masked_rmse,
                            }) # 指标函数，默认：MAE、MSE、RMSE、MAPE、WAPE
CFG.METRICS.TARGET = 'MAE' # 目标指标，用于保存最佳检查点。
CFG.METRICS.BEST = 'min' # 最佳指标，用于保存最佳检查点。'min' 或 'max'。默认值：'min'。如果是 'max'，则指标值越大越好。
CFG.METRICS.NULL_VAL = NULL_VAL # 指标的空值。默认值：np.nan

############################## 训练配置 ##############################

CFG.TRAIN = EasyDict() # 训练设置，必须为训练指定。

# 训练参数
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
 # 保存检查点的目录。默认值：'checkpoints/{MODEL_NAME}/{DATASET_NAME}_{NUM_EPOCHS}_{INPUT_LEN}_{OUTPUT_LEN}'
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
# 检查点保存策略。`CFG.TRAIN.CKPT_SAVE_STRATEGY` 可以是 None、整数值、列表或元组。默认值：None。
# None：每个 epoch 移除最后一个检查点文件。整数：每隔 `CFG.TRAIN.CKPT_SAVE_STRATEGY` 个 epoch 保存一次检查点。
# 列表或元组：当 epoch 在 `CFG.TRAIN.CKPT_SAVE_STRATEGY` 中时保存检查点，当 last_epoch 不在 ckpt_save_strategy 中时移除最后一个检查点文件。
# “移除”操作是将最后一个检查点文件重命名为 .bak 文件，BasicTS会每个10个epoch清空一次.bak文件。
CFG.TRAIN.CKPT_SAVE_STRATEGY = None
CFG.TRAIN.FINETUNE_FROM = None # 微调的检查点路径。默认值：None。如果未指定，模型将从头开始训练。
CFG.TRAIN.STRICT_LOAD = True # 是否严格加载检查点。默认值：True。

# 损失函数
CFG.TRAIN.LOSS = masked_mae # 损失函数，必须为训练指定。

# 优化器设置
CFG.TRAIN.OPTIM = EasyDict() # 优化器设置，必须为训练指定。
CFG.TRAIN.OPTIM.TYPE = 'Adam' # 优化器类型，必须为训练指定。
CFG.TRAIN.OPTIM.PARAM = {
                            'lr': 0.002,
                            'weight_decay': 0.0001,
                        } # 优化器参数

# 学习率调度器设置
CFG.TRAIN.LR_SCHEDULER = EasyDict() # 学习率调度器设置。默认值：None。如果未指定，训练期间不会调整学习率。
CFG.TRAIN.LR_SCHEDULER.TYPE = 'MultiStepLR'

# 学习率调度器类型。
CFG.TRAIN.LR_SCHEDULER.PARAM = {
                            'milestones': [1, 50, 80],
                            'gamma': 0.5
                        } # 学习率调度器参数

# 提前停止
CFG.TRAIN.EARLY_STOPPING_PATIENCE = None # 提前停止的耐心值。默认值：None。如果未指定，则不会使用提前停止。

# 梯度剪裁设置
CFG.TRAIN.CLIP_GRAD_PARAM = None # 梯度剪裁参数（torch.nn.utils.clip_grad_norm_）。默认值：None。如果未指定，则不会使用梯度剪裁。

# 课程学习设置
CFG.TRAIN.CL = EasyDict() # 课程学习设置。默认值：None。如果未指定，则不会使用课程学习。
CFG.TRAIN.CL.CL_EPOCHS = 1 # 每个课程学习阶段的 epoch 数，若指定 CFG.TRAIN.CL，必须指定该参数。
CFG.TRAIN.CL.WARM_EPOCHS = 0 # 热身 epoch 数。默认值：0
CFG.TRAIN.CL.PREDICTION_LENGTH = OUTPUT_LEN # 总预测长度，若指定 CFG.TRAIN.CL，必须指定该参数。
CFG.TRAIN.CL.STEP_SIZE = 1 # 课程学习的步长。默认值：1。当前预测长度将在每个阶段增加 CFG.TRAIN.CL.STEP_SIZE。

# 训练数据加载器设置
CFG.TRAIN.DATA = EasyDict() # 训练数据加载器设置，必须为训练指定。
CFG.TRAIN.DATA.PREFETCH = False # 是否使用预取的数据加载器。详见 https://github.com/justheuristic/prefetch_generator。默认值：False。
CFG.TRAIN.DATA.BATCH_SIZE = 64 # 训练的批量大小。默认值：1
CFG.TRAIN.DATA.SHUFFLE = True # 是否对训练数据进行洗牌。默认值：False
CFG.TRAIN.DATA.COLLATE_FN = None # 训练数据加载器的合并函数。默认值：None
CFG.TRAIN.DATA.NUM_WORKERS = 0 # 训练数据加载器的工作线程数。默认值：0
CFG.TRAIN.DATA.PIN_MEMORY = False # 训练数据加载器是否固定内存。默认值：False

############################## 验证配置 ##############################

CFG.VAL = EasyDict()

# 验证参数
CFG.VAL.INTERVAL = 1 # 每隔 `CFG.VAL.INTERVAL` 个 epoch 进行验证。默认值：1
CFG.VAL.DATA = EasyDict() # 参见 CFG.TRAIN.DATA
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.BATCH_SIZE = 64
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.COLLATE_FN = None
CFG.VAL.DATA.NUM_WORKERS = 0
CFG.VAL.DATA.PIN_MEMORY = False

############################## 测试配置 ##############################

CFG.TEST = EasyDict()

# 测试参数
CFG.TEST.INTERVAL = 1 # 每隔 `CFG.TEST.INTERVAL` 个 epoch 进行测试。默认值：1
CFG.TEST.DATA = EasyDict() # 参见 CFG.TRAIN.DATA
CFG.VAL.DATA.PREFETCH = False
CFG.TEST.DATA.BATCH_SIZE = 64
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.COLLATE_FN = None
CFG.TEST.DATA.NUM_WORKERS = 0
CFG.TEST.DATA.PIN_MEMORY = False

############################## 评估配置 ##############################

CFG.EVAL = EasyDict()

# 评估参数
# 评估时的预测时间范围。默认值为 []。注意：HORIZONS[i] 指的是在 ”第 i 个时间片“ 上进行测试，表示该时间片的损失（Loss）。
# 这是时空预测中的常见配置。对于长序列预测，建议将 HORIZONS 保持为默认值 []，以避免引发误解。
CFG.EVAL.HORIZONS = []
CFG.EVAL.USE_GPU = True # 是否在评估时使用 GPU。默认值：True
CFG.EVAL.SAVE_RESULTS = False # 是否将评估结果保存为一个numpy文件。 默认值：False
