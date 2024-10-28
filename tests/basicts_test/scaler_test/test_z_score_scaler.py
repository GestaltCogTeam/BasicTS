import unittest
import torch
import numpy as np
import os
import json
from basicts.scaler.z_score_scaler import ZScoreScaler

class TestZScoreScaler(unittest.TestCase):

    def setUp(self):
        # Create a mock dataset description and data
        self.dataset_name = 'mock_dataset'
        self.train_ratio = 0.8
        self.norm_each_channel = True
        self.rescale = False

        self._SAMPLES = 100
        self._STEPS = 10
        self._CHANNELS = 5

        # Mock dataset description
        self.description = {
            'shape': (self._SAMPLES, self._STEPS, self._CHANNELS)  # 100 samples, 10 timesteps, 5 channels
        }

        # Create mock dataset directory
        os.makedirs(f'datasets/{self.dataset_name}', exist_ok=True)

        # Mock data
        self.data = np.random.rand(self._SAMPLES, self._STEPS, self._CHANNELS).astype('float32')


        # Save mock description and data to files
        with open(f'datasets/{self.dataset_name}/desc.json', 'w') as f:
            json.dump(self.description, f)
        np.memmap(f'datasets/{self.dataset_name}/data.dat', dtype='float32', mode='w+', shape=self.data.shape)[:] = self.data[:]

        # Initialize the ZScoreScaler
        self.scaler = ZScoreScaler(self.dataset_name, self.train_ratio, self.norm_each_channel, self.rescale)

    def test_transform(self):
        # Create a mock input tensor
        input_data = torch.tensor(self.data[:int(self._SAMPLES*self.train_ratio)], dtype=torch.float32)

        # Apply the transform
        transformed_data = self.scaler.transform(input_data)

        # Check if the mean of the transformed data is approximately 0
        self.assertTrue(torch.allclose(torch.mean(transformed_data[..., self.scaler.target_channel], axis=0, keepdims=True), torch.tensor(0.0), atol=1e-6))

        # Check if the std of the transformed data is approximately 1
        self.assertTrue(torch.allclose(torch.std(transformed_data[..., self.scaler.target_channel], axis=0, keepdims=True, unbiased=False), torch.tensor(1.0), atol=1e-6))

    def test_inverse_transform(self):
        # Create a mock input tensor
        input_data = torch.tensor(self.data[:int(self._SAMPLES*self.train_ratio)], dtype=torch.float32)
        raw_data = input_data.clone()

        # Apply the transform
        transformed_data = self.scaler.transform(input_data)

        # Apply the inverse transform
        inverse_transformed_data = self.scaler.inverse_transform(transformed_data)

        # Check if the inverse transformed data is approximately equal to the original data
        self.assertTrue(torch.allclose(inverse_transformed_data, raw_data, atol=1e-6))
    
    def tearDown(self):
        # Remove the mock dataset directory
        os.remove(f'datasets/{self.dataset_name}/desc.json')
        os.remove(f'datasets/{self.dataset_name}/data.dat')
        os.rmdir(f'datasets/{self.dataset_name}')

if __name__ == '__main__':
    unittest.main()