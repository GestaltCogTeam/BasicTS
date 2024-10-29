import unittest
from unittest.mock import patch, call
import sys
import os
from experiments.train import parse_args
from experiments.train import main

# Add the path to the train.py file
sys.path.append(os.path.abspath(__file__ + "/../.."))


class TestTrain(unittest.TestCase):
    @patch('experiments.train.basicts.launch_training')
    @patch('sys.argv', ['train.py', '-c', 'baselines/STID/PEMS04.py', '-g', '0'])
    def test_launch_training_called_with_correct_args(self, mock_launch_training):
        args = parse_args()
        self.assertEqual(args.cfg, 'baselines/STID/PEMS04.py')
        self.assertEqual(args.gpus, '0')

        # Simulate the main function
        main()

        # Check if launch_training was called with the correct arguments
        mock_launch_training.assert_called_once_with('baselines/STID/PEMS04.py', '0', node_rank=0)





if __name__ == '__main__':
    unittest.main()