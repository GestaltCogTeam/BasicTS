import unittest
import torch
import json
import copy
import numpy as np
from basicts.scaler.min_max_scaler import MinMaxScaler
import os

class TestMinMaxScaler(unittest.TestCase):

    def setUp(self):
        # Mock dataset description and data
        self.dataset_name = 'mock_dataset'
        self.train_ratio = 0.8
        self.norm_each_channel = True
        self.rescale = True

        # create mock dataset directory
        os.makedirs(f'datasets/{self.dataset_name}', exist_ok=True)

        # Mock the dataset description and data
        self.description = {'shape': [100, 10, 1]}
        self.data = np.random.rand(100, 10, 1).astype('float32')

        # Save mock description and data to files
        with open(f'datasets/{self.dataset_name}/desc.json', 'w+') as f:
            json.dump(self.description, f)
        np.memmap(f'datasets/{self.dataset_name}/data.dat', dtype='float32', mode='w+', shape=tuple(self.description['shape']))[:] = self.data

        # Initialize the MinMaxScaler
        self.scaler = MinMaxScaler(self.dataset_name, self.train_ratio, self.norm_each_channel, self.rescale)

    def test_transform(self):
        # Create a sample input tensor
        input_data = torch.tensor(self.data[:30], dtype=torch.float32)

        # Apply the transform method
        transformed_data = self.scaler.transform(copy.deepcopy(input_data))

        # Check if the transformed data is within the range [0, 1]
        self.assertTrue(torch.all(transformed_data >= 0))
        self.assertTrue(torch.all(transformed_data <= 1))

        # Check if the shape of the transformed data is the same as the input
        self.assertEqual(transformed_data.shape, input_data.shape)

    def test_inverse_transform(self):
        # Create a sample input tensor
        input_data = torch.tensor(self.data[:30], dtype=torch.float32)

        # Apply the transform method
        transformed_data = self.scaler.transform(copy.deepcopy(input_data))

        # Apply the inverse_transform method
        inverse_transformed_data = self.scaler.inverse_transform(transformed_data)

        # Check if the inverse transformed data is close to the original data
        self.assertTrue(torch.allclose(inverse_transformed_data, input_data, atol=1e-5))

        # Check if the shape of the inverse transformed data is the same as the input
        self.assertEqual(inverse_transformed_data.shape, input_data.shape)

    def tearDown(self):
        # Clean up the mock files
        os.remove(f'datasets/{self.dataset_name}/desc.json')
        os.remove(f'datasets/{self.dataset_name}/data.dat')
        os.rmdir(f'datasets/{self.dataset_name}')

if __name__ == '__main__':
    unittest.main()
