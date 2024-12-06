import copy
import unittest
from unittest.mock import MagicMock, patch

from easydict import EasyDict

from basicts.launcher import (evaluation_func, launch_evaluation,
                              launch_training)


class TestLauncher(unittest.TestCase):
    """
    Test cases for the launcher.
    """

    @patch('basicts.launcher.get_logger')
    @patch('basicts.launcher.os.path.exists')
    @patch('basicts.launcher.init_cfg')
    @patch('basicts.launcher.set_device_type')
    @patch('basicts.launcher.set_visible_devices')
    @patch('basicts.launcher.evaluation_func')
    def test_launch_evaluation(self, mock_evaluation_func, mock_set_visible_devices, mock_set_device_type, mock_init_cfg, mock_path_exists, mock_get_logger):
        # Mocking
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_path_exists.return_value = True
        mock_init_cfg.return_value = EasyDict({'RUNNER': MagicMock()})

        # Test data
        cfg = 'path/to/config'
        ckpt_path = 'path/to/checkpoint'
        device_type = 'gpu'
        gpus = '0'
        batch_size = 32

        # Call the function
        launch_evaluation('././'+cfg, '././'+ckpt_path, device_type, gpus, batch_size)

        # Assertions
        mock_get_logger.assert_called_once_with('easytorch-launcher')
        mock_logger.info.assert_called_with('Launching EasyTorch evaluation.')
        mock_init_cfg.assert_called_once_with(cfg, save=True)
        mock_set_device_type.assert_called_once_with(device_type)
        mock_set_visible_devices.assert_called_once_with(gpus)
        mock_evaluation_func.assert_called_once_with(mock_init_cfg.return_value, ckpt_path, batch_size)

    @patch('basicts.launcher.easytorch.launch_training')
    def test_launch_training(self, mock_launch_training):
        # Test data
        cfg = 'path/to/config'
        gpus = '0'
        node_rank = 0

        # Call the function
        launch_training('././'+cfg, gpus, node_rank)

        # Assertions
        mock_launch_training.assert_called_once_with(cfg=cfg, devices=gpus, node_rank=node_rank)

    @patch('basicts.launcher.get_logger')
    @patch('basicts.launcher.os.path.exists')
    def test_evaluation_func(self, mock_path_exists, mock_get_logger):
        # Mocking
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_runner = MagicMock()
        mock_path_exists.side_effect = lambda path: path == 'path/to/checkpoint'

        # Test data
        cfg = EasyDict({'RUNNER': MagicMock(return_value=mock_runner)})
        cfg.TEST = EasyDict()
        cfg.TEST.DATA = EasyDict()
        ckpt_path = 'path/to/checkpoint'
        batch_size = 32
        strict = True
        test_cfg = copy.deepcopy(cfg)
        test_cfg.TEST.DATA.BATCH_SIZE = batch_size

        # Call the function
        evaluation_func(cfg, ckpt_path, batch_size, strict)

        # Assertions
        mock_get_logger.assert_called_once_with('easytorch-launcher')
        mock_logger.info.assert_any_call(f"Initializing runner '{cfg['RUNNER']}'")
        mock_runner.init_logger.assert_called_once_with(logger_name='easytorch-evaluation', log_file_name='evaluation_log')
        mock_runner.load_model.assert_called_once_with(ckpt_path=ckpt_path, strict=strict)
        mock_runner.test_pipeline.assert_called_once_with(cfg=test_cfg, save_metrics=True, save_results=True)

if __name__ == '__main__':
    unittest.main()
