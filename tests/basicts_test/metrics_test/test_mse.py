import torch
import numpy as np
import unittest
from basicts.metrics.mse import masked_mse

class TestMaskedMSE(unittest.TestCase):
    def test_masked_mse_no_nulls(self):
        prediction = torch.tensor([1.0, 3.0, 3.0, 5.0])
        target = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = masked_mse(prediction, target)
        expected = torch.tensor(0.5)  # (0 + 1 + 0 + 1) / 4 = 0.5
        self.assertTrue(torch.allclose(result, expected), f"Expected {expected}, but got {result}")

    def test_masked_mse_with_nulls(self):
        prediction = torch.tensor([2.0, 3.0, 3.0, 5.0])
        target = torch.tensor([1.0, 2.0, np.nan, 4.0])
        result = masked_mse(prediction, target)
        expected = torch.tensor(1.0)  # (1 + 1 + 0 + 1) / 3 = 1
        self.assertTrue(torch.allclose(result, expected), f"Expected {expected}, but got {result}")

    def test_masked_mse_with_non_nan_nulls(self):
        prediction = torch.tensor([2.0, 3.0, 3.0, 5.0])
        target = torch.tensor([1.0, 2.0, -1.0, 4.0])
        result = masked_mse(prediction, target, -1.0)
        expected = torch.tensor(1.0)  # (1 + 1 + 0 + 1) / 3 = 1
        self.assertTrue(torch.allclose(result, expected), f"Expected {expected}, but got {result}")

    def test_masked_mse_with_all_nulls(self):
        prediction = torch.tensor([1.0, 2.0, 3.0, 4.0])
        target = torch.tensor([np.nan, np.nan, np.nan, np.nan])
        result = masked_mse(prediction, target)
        expected = torch.tensor(0.0)
        self.assertTrue(torch.allclose(result, expected), f"Expected {expected}, but got {result}")

if __name__ == '__main__':
    unittest.main()