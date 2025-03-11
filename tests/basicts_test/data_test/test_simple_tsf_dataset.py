# pylint: disable=unused-argument
import json
import unittest
from unittest.mock import mock_open, patch

import numpy as np

from basicts.data.simple_tsf_dataset import TimeSeriesForecastingDataset


class TestTimeSeriesForecastingDataset(unittest.TestCase):
    """
    Test the TimeSeriesForecastingDataset class.
    """

    def setUp(self):
        self.dataset_name = 'test_dataset'
        self.train_val_test_ratio = [0.6, 0.2, 0.2]
        self.input_len = 10
        self.output_len = 5
        self.mode = 'train'
        self.overlap = False
        self.logger = None

        self.description = {
            'shape': [100,]
        }
        self.data = np.arange(100, dtype='float32')

        # Mock the file paths
        self.data_file_path = f'datasets/{self.dataset_name}/data.dat'
        self.description_file_path = f'datasets/{self.dataset_name}/desc.json'

    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps({'shape': [100,]}))
    @patch('numpy.memmap')
    def test_load_description(self, mock_memmap, mocked_open):
        mock_memmap.return_value = self.data

        dataset = TimeSeriesForecastingDataset(
            dataset_name=self.dataset_name,
            train_val_test_ratio=self.train_val_test_ratio,
            mode=self.mode,
            input_len=self.input_len,
            output_len=self.output_len,
            overlap=self.overlap,
            logger=self.logger
        )

        self.assertEqual(dataset.description, self.description)

    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_load_description_file_not_found(self, mocked_open):
        with self.assertRaises(FileNotFoundError):
            TimeSeriesForecastingDataset(
                dataset_name=self.dataset_name+'nonexistent',
                train_val_test_ratio=self.train_val_test_ratio,
                mode=self.mode,
                input_len=self.input_len,
                output_len=self.output_len,
                overlap=self.overlap,
                logger=self.logger
            )

    @patch('builtins.open', new_callable=mock_open, read_data='not a json')
    def test_load_description_json_decode_error(self, mocked_open):
        with self.assertRaises(ValueError):
            TimeSeriesForecastingDataset(
            dataset_name=self.dataset_name,
            train_val_test_ratio=self.train_val_test_ratio,
            mode=self.mode,
            input_len=self.input_len,
            output_len=self.output_len,
            overlap=self.overlap,
            logger=self.logger
        )

    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps({'shape': [100,]}))
    @patch('numpy.memmap', side_effect=FileNotFoundError)
    def test_load_data_file_not_found(self, mock_memmap, mocked_open):
        with self.assertRaises(ValueError):
            TimeSeriesForecastingDataset(
                dataset_name=self.dataset_name,
                train_val_test_ratio=self.train_val_test_ratio,
                mode=self.mode,
                input_len=self.input_len,
                output_len=self.output_len,
                overlap=self.overlap,
                logger=self.logger
            )

    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps({'shape': [100,]}))
    @patch('numpy.memmap', side_effect=ValueError)
    def test_load_data_value_error(self, mock_memmap, mocked_open):
        with self.assertRaises(ValueError):
            TimeSeriesForecastingDataset(
                dataset_name=self.dataset_name,
                train_val_test_ratio=self.train_val_test_ratio,
                mode=self.mode,
                input_len=self.input_len,
                output_len=self.output_len,
                overlap=self.overlap,
                logger=self.logger
            )


    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps({'shape': [100,]}))
    @patch('numpy.memmap')
    def test_load_data_train_mode(self, mock_memmap, mocked_open):
        mock_memmap.return_value = self.data

        dataset = TimeSeriesForecastingDataset(
            dataset_name=self.dataset_name,
            train_val_test_ratio=self.train_val_test_ratio,
            mode='train',
            input_len=self.input_len,
            output_len=self.output_len,
            overlap=self.overlap,
            logger=self.logger
        )

        total_len = len(self.data)
        valid_len = int(total_len * self.train_val_test_ratio[1])
        test_len = int(total_len * self.train_val_test_ratio[2])
        expected_data_len = total_len - valid_len - test_len
        self.assertEqual(len(dataset.data), expected_data_len)

    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps({'shape': [100,]}))
    @patch('numpy.memmap')
    def test_load_data_train_mode_overlap(self, mock_memmap, mocked_open):
        mock_memmap.return_value = self.data

        dataset = TimeSeriesForecastingDataset(
            dataset_name=self.dataset_name,
            train_val_test_ratio=self.train_val_test_ratio,
            mode='train',
            input_len=self.input_len,
            output_len=self.output_len,
            overlap=True,
            logger=self.logger
        )

        total_len = len(self.data)
        valid_len = int(total_len * self.train_val_test_ratio[1])
        test_len = int(total_len * self.train_val_test_ratio[2])
        expected_data_len = total_len - valid_len - test_len + self.output_len
        self.assertEqual(len(dataset.data), expected_data_len)

    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps({'shape': [100,]}))
    @patch('numpy.memmap')
    def test_load_data_valid_mode(self, mock_memmap, mocked_open):
        mock_memmap.return_value = self.data

        dataset = TimeSeriesForecastingDataset(
            dataset_name=self.dataset_name,
            train_val_test_ratio=self.train_val_test_ratio,
            mode='valid',
            input_len=self.input_len,
            output_len=self.output_len,
            overlap=self.overlap,
            logger=self.logger
        )

        valid_len = int(len(self.data) * self.train_val_test_ratio[1])
        expected_data_len = valid_len
        self.assertEqual(len(dataset.data), expected_data_len)

    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps({'shape': [100,]}))
    @patch('numpy.memmap')
    def test_load_data_valid_mode_overlap(self, mock_memmap, mocked_open):
        mock_memmap.return_value = self.data

        dataset = TimeSeriesForecastingDataset(
            dataset_name=self.dataset_name,
            train_val_test_ratio=self.train_val_test_ratio,
            mode='valid',
            input_len=self.input_len,
            output_len=self.output_len,
            overlap=True,
            logger=self.logger
        )

        valid_len = int(len(self.data) * self.train_val_test_ratio[1])
        expected_data_len = valid_len + self.input_len - 1 + self.output_len
        self.assertEqual(len(dataset.data), expected_data_len)


    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps({'shape': [100,]}))
    @patch('numpy.memmap')
    def test_load_data_test_mode(self, mock_memmap, mocked_open):
        mock_memmap.return_value = self.data

        dataset = TimeSeriesForecastingDataset(
            dataset_name=self.dataset_name,
            train_val_test_ratio=self.train_val_test_ratio,
            mode='test',
            input_len=self.input_len,
            output_len=self.output_len,
            overlap=self.overlap,
            logger=self.logger
        )

        test_len = int(len(self.data) * self.train_val_test_ratio[1])
        expected_data_len = test_len
        self.assertEqual(len(dataset.data), expected_data_len)


    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps({'shape': [100,]}))
    @patch('numpy.memmap')
    def test_load_data_test_mode_overlap(self, mock_memmap, mocked_open):
        mock_memmap.return_value = self.data

        dataset = TimeSeriesForecastingDataset(
            dataset_name=self.dataset_name,
            train_val_test_ratio=self.train_val_test_ratio,
            mode='test',
            input_len=self.input_len,
            output_len=self.output_len,
            overlap=True,
            logger=self.logger
        )


        test_len = int(len(self.data) * self.train_val_test_ratio[2])
        expected_data_len = test_len + self.input_len - 1
        self.assertEqual(len(dataset.data), expected_data_len)


    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps({'shape': [100,]}))
    @patch('numpy.memmap')
    def test_getitem(self, mock_memmap, mocked_open):
        mock_memmap.return_value = self.data

        dataset = TimeSeriesForecastingDataset(
            dataset_name=self.dataset_name,
            train_val_test_ratio=self.train_val_test_ratio,
            mode=self.mode,
            input_len=self.input_len,
            output_len=self.output_len,
            overlap=self.overlap,
            logger=self.logger
        )

        sample = dataset[0]
        expected_inputs = np.arange(self.input_len, dtype='float32')
        expected_target = np.arange(self.input_len, self.input_len + self.output_len, dtype='float32')

        np.testing.assert_array_equal(sample['inputs'], expected_inputs)
        np.testing.assert_array_equal(sample['target'], expected_target)

    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps({'shape': [100,]}))
    @patch('numpy.memmap')
    def test_len(self, mock_memmap, mocked_open):
        mock_memmap.return_value = self.data

        dataset = TimeSeriesForecastingDataset(
            dataset_name=self.dataset_name,
            train_val_test_ratio=self.train_val_test_ratio,
            mode=self.mode,
            input_len=self.input_len,
            output_len=self.output_len,
            overlap=self.overlap,
            logger=self.logger
        )


        expected_len = len(self.data)*self.train_val_test_ratio[0] - self.input_len - self.output_len + 1
        self.assertEqual(len(dataset), expected_len)


if __name__ == '__main__':
    unittest.main()
