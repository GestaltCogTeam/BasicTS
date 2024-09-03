import os

import torch
import numpy as np

PROJECT_DIR = os.path.abspath(__file__ + '/../../../..')
os.chdir(PROJECT_DIR)


# hyper parameterts
duration = 10000  # time series length

def generate_gaussian_noise_sequence():
    x = np.arange(0, duration, 1)
    y = np.random.normal(0, 1, duration)
    return x, y

# generate gaussian sequence
time_points, gaussian_noise_sequence = generate_gaussian_noise_sequence()

# save pulse sequence
data = torch.Tensor(gaussian_noise_sequence).unsqueeze(-1).unsqueeze(-1).numpy()
# mkdir datasets/raw_data/Gaussian
if not os.path.exists('datasets/raw_data/Gaussian'):
    os.makedirs('datasets/raw_data/Gaussian')
np.save('datasets/raw_data/Gaussian/Gaussian.npy', data)
