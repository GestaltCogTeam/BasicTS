import unittest

import numpy as np
import torch

from basicts.metrics.mae import masked_mae


class TestMaskedMAE(unittest.TestCase):
    """
    Test the masked_mae function from basicts.metrics.mae.
    """

    def test_masked_mae_no_nulls(self):
        prediction = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        result = masked_mae(prediction, target)
        expected = torch.tensor(0.0)
        self.assertTrue(torch.allclose(result, expected), f"Expected {expected}, but got {result}")

    def test_masked_mae_with_nulls(self):
        prediction = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, np.nan, 3.0])
        result = masked_mae(prediction, target)
        expected = torch.tensor(0.0)
        self.assertTrue(torch.allclose(result, expected), f"Expected {expected}, but got {result}")

    def test_masked_mae_with_nulls_and_differences(self):
        prediction = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, np.nan, 4.0])
        result = masked_mae(prediction, target)
        expected = torch.tensor(0.5)
        self.assertTrue(torch.allclose(result, expected), f"Expected {expected}, but got {result}")

    def test_masked_mae_with_custom_null_val(self):
        prediction = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, -1.0, 4.0])
        result = masked_mae(prediction, target, null_val=-1.0)
        expected = torch.tensor(0.5)
        self.assertTrue(torch.allclose(result, expected), f"Expected {expected}, but got {result}")

    def test_masked_mae_all_nulls(self):
        prediction = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([np.nan, np.nan, np.nan])
        result = masked_mae(prediction, target)
        expected = torch.tensor(0.0)  # Since all are nulls, the MAE should be zero
        self.assertTrue(torch.allclose(result, expected), f"Expected {expected}, but got {result}")

if __name__ == "__main__":
    unittest.main()
