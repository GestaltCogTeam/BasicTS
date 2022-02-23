import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from operator import index
from basicts.data.base_dataset import BaseDataset

dataset_name  = 'METR-LA'
raw_file_path = 'datasets/{0}/data.pkl'.format(dataset_name)
index_file_path = 'datasets/{0}/index.pkl'.format(dataset_name)
mode = 'train'
dataset = BaseDataset(raw_file_path, index_file_path, mode)
for _ in dataset:
    data = _
    a = 1
    break