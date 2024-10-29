import unittest
from unittest.mock import patch
from experiments.evaluate import parse_args

class TestEvaluate(unittest.TestCase):

    @patch('sys.argv', ['evaluate.py'])
    def test_default_args(self):
        args = parse_args()
        self.assertEqual(args.config, "baselines/STID/PEMS08_LTSF.py")
        self.assertEqual(args.checkpoint, "checkpoints/STID/PEMS08_100_336_336/97d131cadc14bd2b9ffa892d59d55129/STID_best_val_MAE.pt")
        self.assertEqual(args.gpus, "5")
        self.assertEqual(args.device_type, "gpu")
        self.assertIsNone(args.batch_size)

    @patch('sys.argv', ['evaluate.py', '-cfg', 'custom_config.py', '-ckpt', 'custom_checkpoint.pt', '-g', '0', '-d', 'cpu', '-b', '32'])
    def test_custom_args(self):
        args = parse_args()
        self.assertEqual(args.config, "custom_config.py")
        self.assertEqual(args.checkpoint, "custom_checkpoint.pt")
        self.assertEqual(args.gpus, "0")
        self.assertEqual(args.device_type, "cpu")
        self.assertEqual(args.batch_size, '32')



if __name__ == '__main__':
    unittest.main()