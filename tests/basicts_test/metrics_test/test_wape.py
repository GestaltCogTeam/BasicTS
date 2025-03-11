import unittest

import numpy as np
import torch

from basicts.metrics.wape import masked_wape


class TestMaskedWape(unittest.TestCase):
    """
    Test the masked WAPE function.
    """

    def test_masked_wape_basic(self):
        prediction = torch.tensor([[2.0, 2.0, 3.0], [6.0, 5.0, 7.0]])
        target = torch.tensor([[1.0, 2.0, 2.0], [4.0, 5.0, 6.0]])
        result = masked_wape(prediction, target)
        expected = torch.tensor(0.3) #(0.4 + 0.2) / 2 = 0.3
        self.assertTrue(torch.allclose(result, expected), f"Expected {expected}, but got {result}")

    def test_masked_wape_with_nulls(self):
        prediction = torch.tensor([[2.0, 2.0, 4.0], [8.0, 5.0, 6.0]])
        target = torch.tensor([[1.0, np.nan, 3.0], [5.0, 5.0, np.nan]])
        result = masked_wape(prediction, target)
        expected = torch.tensor(0.4) # (0.5 + 0.3) / 2  = 0.4
        self.assertTrue(torch.allclose(result, expected), f"Expected {expected}, but got {result}")

    def test_masked_wape_with_custom_null_val(self):
        prediction = torch.tensor([[2.0, 2.0, 4.0], [8.0, 5.0, 6.0]])
        target = torch.tensor([[1.0, -1.0, 3.0], [5.0, 5.0, -1.0]])
        result = masked_wape(prediction, target, null_val=-1.0)
        expected = torch.tensor(0.4) # (0.5 + 0.3) / 2  = 0.4
        self.assertTrue(torch.allclose(result, expected), f"Expected {expected}, but got {result}")

    def test_masked_wape_with_all_null_vals(self):
        prediction = torch.tensor([[3.0, 2.0, 5.0], [4.0, 5.0, 6.0]])
        target = torch.tensor([[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]])
        result = masked_wape(prediction, target, null_val=-1.0)
        expected = torch.tensor(0.0) # No valid entries, should return 0
        self.assertTrue(torch.allclose(result, expected), f"Expected {expected}, but got {result}")

if __name__ == "__main__":
    unittest.main()
