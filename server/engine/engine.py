#coding:utf-8
import os
import sys
from typing import Optional, Union

import easytorch
from easytorch.config import init_cfg
from easytorch.device import set_device_type
from easytorch.utils import set_visible_devices

sys.path.append(os.path.abspath(os.path.join(__file__,'..','..','..')))


class inference_engine(object):
    """
    Inference engine for EasyTorch.
    """

    def __init__(self, cfg_path: str, ckpt_path: str, device_type: str = 'gpu', gpus: Optional[str] = None,\
                 context_length: int = 72, prediction_length: int = 36):
        """
        Initializes the inference engine.

        Args:
            cfg_path (str): Path to the EasyTorch configuration file.
            ckpt_path (str): Path to the model checkpoint.
            device_type (str, optional): Device type to use ('cpu' or 'gpu'). Defaults to 'gpu'.
            gpus (Optional[str]): GPU device IDs to use. Defaults to None (use all available GPUs).
        """
        self.cfg_path = cfg_path
        self.ckpt_path = ckpt_path
        self.device_type = device_type
        self.gpus = gpus
        self.context_length = context_length
        self.prediction_length = prediction_length

        self.cfg_dict = init_cfg(self.cfg_path, save=True)

        set_device_type(self.device_type)
        if self.device_type != 'cpu':
            set_visible_devices(self.gpus)

        self.runner = self.load_runner()

    def load_runner(self) -> easytorch.Runner:
        cfg = self.cfg_dict
        # Load the runner
        runner = cfg['RUNNER'](cfg)
        # initialize the logger for the runner
        runner.init_logger(logger_name='easytorch-inference', log_file_name='inference_log')

        # setup the graph if needed
        if runner.need_setup_graph:
            runner.setup_graph(cfg=cfg, train=False)
        runner.load_model(ckpt_path=self.ckpt_path, strict=True)

        return runner

    def inference(self, input_data: Union[list, str]) -> tuple:
        if self.runner is None:
            raise ValueError('Runner is not loaded. Please load the runner first.')
        result = self.runner.inference_pipeline(cfg=self.cfg_dict, input_data=input_data, output_data_file_path='',
                                                context_length=self.context_length, prediction_length=self.prediction_length)
        self.runner.meter_pool = None
        return result


