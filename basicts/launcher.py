import os
from typing import Dict, Union, Optional
import traceback
import easytorch
from easytorch.utils import get_logger, set_visible_devices
from easytorch.config import init_cfg
from easytorch.device import set_device_type

def evaluation_func(cfg: Dict,
                    ckpt_path: str = None,
                    batch_size: Optional[int] = None,
                    strict: bool = True) -> None:
    """
    Starts the evaluation process.

    This function performs the following steps:
    1. Initializes the runner specified in the configuration (`cfg`).
    2. Sets up logging for the evaluation process.
    3. Loads the model checkpoint.
    4. Executes the test pipeline using the initialized runner.

    Args:
        cfg (Dict): EasyTorch configuration dictionary.
        ckpt_path (str): Path to the model checkpoint. If not provided, the best model checkpoint is loaded automatically.
        batch_size (Optional[int]): Batch size for evaluation. If not specified, 
                                    it should be defined in the config. Defaults to None.
        strict (bool): Enforces that the checkpoint keys match the model. Defaults to True.

    Raises:
        Exception: Catches any exception, logs the traceback, and re-raises it.
    """

    # initialize the runner
    logger = get_logger('easytorch-launcher')
    logger.info(f"Initializing runner '{cfg['RUNNER']}'")
    runner = cfg['RUNNER'](cfg)

    # initialize the logger for the runner
    runner.init_logger(logger_name='easytorch-evaluation', log_file_name='evaluation_log')

    try:
        # set batch size if provided
        if batch_size is not None:
            cfg.TEST.DATA.BATCH_SIZE = batch_size
        else:
            assert 'BATCH_SIZE' in cfg.TEST.DATA, 'Batch size must be specified either in the config or as an argument.'

        # load the model checkpoint
        if ckpt_path is None or not os.path.exists(ckpt_path):
            ckpt_path_auto = os.path.join(runner.ckpt_save_dir, '{}_best_val_{}.pt'.format(runner.model_name, runner.target_metrics.replace('/', '_')))
            logger.info(f'Checkpoint file not found at {ckpt_path}. Loading the best model checkpoint `{ckpt_path_auto}` automatically.')
            if not os.path.exists(ckpt_path_auto):
                raise FileNotFoundError(f'Checkpoint file not found at {ckpt_path}')
            runner.load_model(ckpt_path=ckpt_path_auto, strict=strict)
        else:
            logger.info(f'Loading model checkpoint from {ckpt_path}')
            runner.load_model(ckpt_path=ckpt_path, strict=strict)

        # start the evaluation pipeline
        runner.test_pipeline(cfg=cfg, save_metrics=True, save_results=True)

    except BaseException as e:
        # log the exception and re-raise it
        runner.logger.error(traceback.format_exc())
        raise e

def launch_evaluation(cfg: Union[Dict, str],
                      ckpt_path: str,
                      device_type: str = 'gpu',
                      gpus: Optional[str] = None,
                      batch_size: Optional[int] = None) -> None:
    """
    Launches the evaluation process using EasyTorch.

    Args:
        cfg (Union[Dict, str]): EasyTorch configuration as a dictionary or a path to a config file.
        ckpt_path (str): Path to the model checkpoint.
        device_type (str, optional): Device type to use ('cpu' or 'gpu'). Defaults to 'gpu'.
        gpus (Optional[str]): GPU device IDs to use. Defaults to None (use all available GPUs).
        batch_size (Optional[int]): Batch size for evaluation. Defaults to None (use value from config).

    Raises:
        AssertionError: If the batch size is not specified in either the config or as an argument.
    """

    logger = get_logger('easytorch-launcher')
    logger.info('Launching EasyTorch evaluation.')

    # initialize the configuration
    cfg = init_cfg(cfg, save=True)

    # set the device type (CPU, GPU, or MLU)
    set_device_type(device_type)

    # set the visible GPUs if the device type is not CPU
    if device_type != 'cpu':
        set_visible_devices(gpus)

    # run the evaluation process
    evaluation_func(cfg, ckpt_path, batch_size)

def launch_training(cfg: Union[Dict, str],
                    gpus: Optional[str] = None,
                    node_rank: int = 0) -> None:
    """
    Launches the training process using EasyTorch.

    Args:
        cfg (Union[Dict, str]): EasyTorch configuration as a dictionary or a path to a config file.
        gpus (Optional[str]): GPU device IDs to use. Defaults to None (use all available GPUs).
        node_rank (int, optional): Rank of the current node in distributed training. Defaults to 0.
    """

    # placeholder for potential pre-processing steps (e.g., model registration, config validation)
    pass

    # launch the training process
    easytorch.launch_training(cfg=cfg, devices=gpus, node_rank=node_rank)
