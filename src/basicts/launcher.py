import traceback
from typing import Optional

from easytorch.device import set_device_type
from easytorch.launcher.dist_wrap import dist_wrap
from easytorch.utils import get_logger, set_visible_devices

from basicts.configs.base_config import BasicTSConfig
from basicts.runners import BasicTSRunner


class BasicTSLauncher:

    """
    BasicTSLauncher to launch the training and evaluation process.
    """

    @staticmethod
    def launch_training(cfg: BasicTSConfig, node_rank: int = 0) -> None:
        """
        Launches the training process.

        This method initializes the runner specified in the configuration, sets up logging,
        and starts the training loop.

        Args:
            cfg (BasicTSConfig): configuration dictionary.
        """
        # launch the training process
        logger = get_logger("BasicTS-launcher")
        logger.info("Launching BasicTS training.")

        if node_rank == 0:
            cfg.save()

        if cfg.get("gpus", None) is not None or cfg.get("mlus", None) is not None:
            if cfg.get("gpus", None) is not None and cfg.get("mlus", None) is None:
                set_device_type("gpu")
                device_num = cfg.get("gpu_num", 0)
            elif cfg.get("gpus", None) is None and cfg.get("mlus", None) is not None:
                set_device_type("mlu")
                device_num = cfg.get("mlu_num", 0)
            else:
                raise ValueError("At least one of `CFG.GPU_NUM` and `CFG.MLU_NUM` is 0.")
            set_visible_devices(cfg.get("gpus", None))
        else:
            set_device_type("cpu")
            device_num = 0

        train_dist = dist_wrap(
            training_func,
            node_num=cfg.get("dist_node_num", 1),
            device_num=device_num,
            node_rank=node_rank,
            dist_backend=cfg.get("dist_backend"),
            init_method=cfg.get("dist_init_method")
        )
        train_dist(cfg)

    @staticmethod
    def launch_evaluation(
        cfg: BasicTSConfig,
        ckpt_path: str,
        gpus: Optional[str] = None,
        batch_size: Optional[int] = None
        ) -> None:
        """
        Launches the evaluation process.

        This method initializes the runner specified in the configuration, sets up logging,
        and starts the evaluation loop.

        Args:
            cfg (BasicTSConfig): configuration dictionary.
        """

        logger = get_logger("BasicTS-launcher")
        logger.info("Launching BasicTS evaluation.")

        set_device_type("gpu" if gpus else "cpu")
        # set the visible GPUs if the device type is not CPU
        if gpus:
            set_visible_devices(gpus)

        # TODO: support load config from json file
        # init_cfg(cfg)
        cfg.gpus = gpus
        cfg.gpu_num = len(gpus.split(",")) if gpus else 0
        # set batch size if provided
        if batch_size is not None:
            cfg.test_batch_size = batch_size

        # initialize the runner
        runner = BasicTSRunner(cfg)

        # initialize the logger for the runner
        runner.init_logger(logger_name="BasicTS-evaluation", log_file_name="evaluation_log")

        # start the evaluation pipeline
        runner.eval(ckpt_path)

def training_func(cfg: BasicTSConfig):
    # init runner
    runner = BasicTSRunner(cfg)
    # init logger (after making ckpt save dir)
    runner.init_logger(logger_name="BasicTS-training", log_file_name="training_log")
    # train
    try:
        runner.train()
    except BaseException as e:
        # log exception to file
        runner.logger.error(traceback.format_exc())
        raise e
