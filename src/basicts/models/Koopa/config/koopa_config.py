from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig

@dataclass
class KoopaConfig(BasicTSModelConfig):
    """
        Config class for Koopa model.
        说明:
            alpha (float): 缩放或权重系数。
            enc_in (int): 输入特征维度。
            input_len (int, optional): 输入序列长度。
            output_len (int, optional): 预测序列长度。
            seg_len (int): 分段长度。
            num_blocks (int): 模块/块数量。
            dynamic_dim (int): 动态特征维度。
            hidden_dim (int): 隐藏层维度。
            hidden_layers (int): 隐藏层数量。
            multistep (bool): 是否为多步预测模式。
        """
    alpha: float = field(default=0.2, metadata={"help": "Scaling coefficient."})
    enc_in: int = field(default=7, metadata={"help": "Input feature dimension."})
    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Prediction length."})
    seg_len: int = field(default=48, metadata={"help": "Segment length. Recommended: e.g., 24 for hourly data."})
    num_blocks: int = field(default=3, metadata={"help": "Number of blocks."})
    dynamic_dim: int = field(default=64, metadata={"help": "Dynamic feature dimension. Must be > 0."})
    hidden_dim: int = field(default=64, metadata={"help": "Hidden dimension."})
    hidden_layers: int = field(default=2, metadata={"help": "Number of hidden layers (>=2 recommended)."})
    multistep: bool = field(default=False, metadata={"help": "Whether to use multistep forecasting."})