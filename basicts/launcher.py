import os
import traceback
from typing import Dict, List, Optional, Union

from easytorch.config import init_cfg
from easytorch.device import set_device_type
from easytorch.launcher.dist_wrap import dist_wrap
from easytorch.utils import get_logger, set_visible_devices

from basicts.configs.base_config import BasicTSConfig


class BasicTSLauncher:

    @classmethod
    def launch_training(cls, cfg: BasicTSConfig, node_rank: int = 0) -> None:
        """
        Launches the training process.

        This method initializes the runner specified in the configuration, sets up logging,
        and starts the training loop.

        Args:
            cfg (Dict): EasyTorch configuration dictionary.
        """
        # launch the training process
        logger = get_logger('BasicTS-launcher')
        logger.info('Launching BasicTS training.')

        if node_rank == 0:
            cfg.save()

        if cfg.get('gpus', None) is not None or cfg.get('mlus', None) is not None:
            if cfg.get('gpus', None) is not None and cfg.get('mlus', None) is None:
                set_device_type('gpu')
                device_num = cfg.get('gpu_num', 0)
            elif cfg.get('gpus', None) is None and cfg.get('mlus', None) is not None:
                set_device_type('mlu')
                device_num = cfg.get('mlu_num', 0)
            else:
                raise ValueError('At least one of `CFG.GPU_NUM` and `CFG.MLU_NUM` is 0.')
            set_visible_devices(cfg.get('gpus', None))
        else:
            set_device_type('cpu')
            device_num = 0

        train_dist = dist_wrap(
            training_func,
            node_num=cfg.get('dist_node_num', 1),
            device_num=device_num,
            node_rank=node_rank,
            dist_backend=cfg.get('dist_backend'),
            init_method=cfg.get('dist_init_method')
        )
        train_dist(cfg)

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

    # setup the graph if needed
    if runner.need_setup_graph:
        runner.setup_graph(cfg=cfg, train=False)

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

    # check params
    # cfg path which start with dot will crash the easytorch, just remove dot
    while isinstance(cfg, str) and cfg.startswith(('./','.\\')):
        cfg = cfg[2:]
    while ckpt_path.startswith(('./','.\\')):
        ckpt_path = ckpt_path[2:]

    # initialize the configuration
    cfg = init_cfg(cfg, save=True)

    # set the device type (CPU, GPU, or MLU)
    set_device_type(device_type)

    # set the visible GPUs if the device type is not CPU
    if device_type != 'cpu':
        set_visible_devices(gpus)

    # run the evaluation process
    evaluation_func(cfg, ckpt_path, batch_size)

def launch_training(args: List[str], node_rank: int = 0) -> None:
    """
    Launches the training process using EasyTorch.

    Args:
        args (List[str]): Command line arguments.
        node_rank (int, optional): Rank of the current node in distributed training. Defaults to 0.
    """

    # placeholder for potential pre-processing steps (e.g., model registration, config validation)

    # cfg path which start with dot will crash the easytorch, just remove dot
    # while isinstance(cfg, str) and cfg.startswith(('./','.\\')):
    #     cfg = cfg[2:]

    # launch the training process
    logger = get_logger('BasicTS-launcher')
    logger.info('Launching BasicTS training.')

    cfg = load_config(args, node_rank == 0)

    if cfg.get('DEVICE') is not None:
        set_device_type(cfg['DEVICE'])
        device_num = cfg.get('DEVICE_NUM', 0)
    elif cfg.gpus is not None or cfg.get('MLU_NUM', 0) != 0:
        if cfg.gpus is not None and cfg.get('MLU_NUM', 0) == 0:
            set_device_type('gpu')
            device_num = cfg.get('GPU_NUM', 0)
        elif cfg.gpus is None and cfg.get('MLU_NUM', 0) != 0:
            set_device_type('mlu')
            device_num = cfg.get('MLU_NUM', 0)
        else:
            raise ValueError('At least one of `CFG.GPU_NUM` and `CFG.MLU_NUM` is 0.')
        set_visible_devices(cfg.gpus)
    else:
        set_device_type('cpu')
        device_num = 0

    train_dist = dist_wrap(
        training_func,
        node_num=cfg.get('DIST_NODE_NUM', 1),
        device_num=device_num,
        node_rank=node_rank,
        dist_backend=cfg.get('DIST_BACKEND'),
        init_method=cfg.get('DIST_INIT_METHOD')
    )
    train_dist(cfg)


def training_func(cfg: BasicTSConfig):
    """Start training

    1. Init runner defined by `cfg`
    2. Init logger
    3. Call `train()` method in the runner

    Args:
        cfg (Dict): Easytorch config.
    """

    # init runner
    logger = get_logger('BasicTS-launcher')
    logger.info('Initializing runner "{}"'.format(cfg.runner))
    runner = cfg.runner(cfg)

    # init logger (after making ckpt save dir)
    runner.init_logger(logger_name='BasicTS-training', log_file_name='training_log')

    # train
    try:
        runner.train(cfg)
    except BaseException as e:
        # log exception to file
        runner.logger.error(traceback.format_exc())
        raise e

def inference_func(cfg: Dict,
                    input_data_file_path: str,
                    output_data_file_path: str,
                    ckpt_path: str,
                    strict: bool = True,
                    context_length: int = 0,
                    prediction_length: int = 0) -> None:
    """
    Starts the inference process.

    This function performs the following steps:
    1. Initializes the runner specified in the configuration (`cfg`).
    2. Sets up logging for the inference process.
    3. Loads the model checkpoint.
    4. Executes the inference pipeline using the initialized runner.

    Args:
        cfg (Dict): EasyTorch configuration dictionary.
        input_data_file_path (str): Path to the input data file.
        output_data_file_path (str): Path to the output data file.
        ckpt_path (str): Path to the model checkpoint. If not provided, the best model checkpoint is loaded automatically.
        strict (bool): Enforces that the checkpoint keys match the model. Defaults to True.
        context_length (int): Context length for inference, only used for utfs models.
        prediction_length (int): Prediction length for inference, only used for utfs models.

    Raises:
        Exception: Catches any exception, logs the traceback, and re-raises it.
    """

    # initialize the runner
    logger = get_logger('easytorch-launcher')
    logger.info(f"Initializing runner '{cfg['RUNNER']}'")
    runner = cfg['RUNNER'](cfg)

    # initialize the logger for the runner
    runner.init_logger(logger_name='easytorch-inference', log_file_name='inference_log')

    # setup the graph if needed
    if runner.need_setup_graph:
        runner.setup_graph(cfg=cfg, train=False)

    try:
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

        # start the inference pipeline
        runner.inference_pipeline(cfg=cfg, input_data=input_data_file_path, output_data_file_path=output_data_file_path, \
                                  context_length=context_length, prediction_length=prediction_length)

    except BaseException as e:
        # log the exception and re-raise it
        runner.logger.error(traceback.format_exc())
        raise e

def launch_inference(cfg: Union[Dict, str],
                      ckpt_path: str,
                      input_data_file_path: str,
                      output_data_file_path: str,
                      device_type: str = 'gpu',
                      gpus: Optional[str] = None,
                      context_length: int = 0,
                      prediction_length: int = 0) -> None:
    """
    Launches the inference process.

    Args:
        cfg (Union[Dict, str]): EasyTorch configuration as a dictionary or a path to a config file.
        ckpt_path (str): Path to the model checkpoint.
        input_data_file_path (str): Path to the input data file.
        output_data_file_path (str): Path to the output data file.
        device_type (str, optional): Device type to use ('cpu' or 'gpu'). Defaults to 'gpu'.
        gpus (Optional[str]): GPU device IDs to use. Defaults to None (use all available GPUs).
        context_length (int): Context length for inference, only used for utfs models.
        prediction_length (int): Prediction length for inference, only used for utfs models.

    Raises:
        AssertionError: If the batch size is not specified in either the config or as an argument.
    """

    logger = get_logger('easytorch-launcher')
    logger.info('Launching EasyTorch inference.')

    # check params
    # cfg path which start with dot will crash the easytorch, just remove dot
    while isinstance(cfg, str) and cfg.startswith(('./','.\\')):
        cfg = cfg[2:]
    while ckpt_path.startswith(('./','.\\')):
        ckpt_path = ckpt_path[2:]

    # initialize the configuration
    cfg_dict = init_cfg(cfg, save=True)

    # set the device type (CPU, GPU, or MLU)
    set_device_type(device_type)

    # set the visible GPUs if the device type is not CPU
    if device_type != 'cpu':
        set_visible_devices(gpus)

    # run the inference process
    inference_func(cfg_dict, input_data_file_path, output_data_file_path, ckpt_path, context_length=context_length, prediction_length=prediction_length)

def import_config(path: str) -> BasicTSConfig:
    """
    Import the configuration from a file.

    Args:
        path (str): Path to the configuration file.
        verbose (bool, optional): Whether to print verbose information. Defaults to False.

    Returns:
        BasicTSConfig: Imported configuration.
    """
    if path.find('.py') != -1:
        path = path[:path.find('.py')].replace('/', '.').replace('\\', '.')
    cfg_name = path.split('.')[-1]
    cfg = __import__(path, fromlist=[cfg_name]).BasicTSForecastingConfig()

    return cfg
