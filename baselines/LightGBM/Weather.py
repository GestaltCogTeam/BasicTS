import os
import sys
sys.path.append(os.path.abspath(__file__ + "/.."))

# from evaluate_ar import evaluate
from evaluate import evaluate

import numpy as np

# construct configs
dataset_name = "Weather"
input_len = 336
output_len = 336
gpu_num = 1
null_val = np.nan
train_data_dir = "datasets/" + dataset_name
rescale = True
batch_size = 128 # only used for collecting data
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(evaluate(project_dir, train_data_dir, input_len, output_len, rescale, null_val, batch_size, patch_len=1))
print(evaluate(project_dir, train_data_dir, input_len, output_len, rescale, null_val, batch_size))
