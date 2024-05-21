import os
import sys
import numpy as np
import matplotlib.pyplot as plt
PROJECT_DIR = os.path.abspath(__file__ + "/../../../..")
os.chdir(PROJECT_DIR)


def generate_gaussian_noise_sequence(duration):
    time_points = np.arange(0, duration, 1)
    gaussion_noise_sequence = np.random.normal(0, 1, duration)
    return time_points, gaussion_noise_sequence

# hyper parameterts
duration = 10000  # time series length

# generate gaussian sequence
time_points, gaussian_noise_sequence = generate_gaussian_noise_sequence(duration)

# save pulse sequence
import torch
data = torch.Tensor(gaussian_noise_sequence).unsqueeze(-1).unsqueeze(-1).numpy()
# mkdir datasets/raw_data/Gaussian
if not os.path.exists('datasets/raw_data/Gaussian'):
    os.makedirs('datasets/raw_data/Gaussian')
np.save('datasets/raw_data/Gaussian/Gaussian.npy', data)
