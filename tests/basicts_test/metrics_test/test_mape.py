import unittest
import torch
import numpy as np
from basicts.metrics.mape import masked_mape

class TestMaskedMAPE(unittest.TestCase):

    def test_basic_functionality(self):
        prediction = torch.tensor([2.0, 3.0, 3.0])
        target = torch.tensor([1.0, 3.0, 2.0])
        result = masked_mape(prediction, target)
        expected = torch.tensor(0.5)  # (1/1 + 0/3 + 1/2) / 3 = 0.5
        self.assertTrue(torch.allclose(result, expected), f"Expected {expected}, but got {result}")

    def test_with_zeros_in_target(self):
        prediction = torch.tensor([2.0, 3.0, 4.0])
        target = torch.tensor([0.0, 3.0, 2.0])
        result = masked_mape(prediction, target)
        expected = torch.tensor(0.5)  # (0/3 + 2/2) / 2 = 0.5
        self.assertTrue(torch.allclose(result, expected), f"Expected {expected}, but got {result}")

    def test_with_null_values(self):
        prediction = torch.tensor([2.0, 3.0, 4.0])
        target = torch.tensor([np.nan, 3.0, 2.0])
        result = masked_mape(prediction, target)
        expected = torch.tensor(0.5)  # (0/3 + 2/2) / 2 = 0.5
        self.assertTrue(torch.allclose(result, expected), f"Expected {expected}, but got {result}")

    def test_with_custom_null_value(self):
        prediction = torch.tensor([2.0, 3.0, 4.0])
        target = torch.tensor([-1.0, 3.0, 2.0])
        result = masked_mape(prediction, target, null_val=-1.0)
        expected = torch.tensor(0.5)  # (0/3 + 2/2) / 2 = 0.5
        self.assertTrue(torch.allclose(result, expected), f"Expected {expected}, but got {result}")

    def test_all_zeros_in_target(self):
        prediction = torch.tensor([2.0, 3.0, 4.0])
        target = torch.tensor([0.0, 0.0, 0.0])
        result = masked_mape(prediction, target)
        expected = torch.tensor(0.0)  # No valid entries, should return 0
        self.assertTrue(torch.allclose(result, expected), f"Expected {expected}, but got {result}")

if __name__ == '__main__':
    unittest.main()