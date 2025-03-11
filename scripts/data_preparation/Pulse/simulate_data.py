import os

import numpy as np
import torch

PROJECT_DIR = os.path.abspath(__file__ + '/../../../..')
os.chdir(PROJECT_DIR)


# hyper parameterts
duration = 20000  # time series length
min_interval = 30  # minimum interval between two pulses
max_interval = 30  # maximum interval between two pulses

def generate_pulse_sequence():
    x = np.arange(0, duration, 1)
    y = np.zeros_like(x)

    current_time = 0
    while current_time < duration:
        pulse_interval = np.random.uniform(min_interval, max_interval)
        pulse_width = 1
        y[int(current_time):int(current_time + pulse_width)] = 1
        current_time += pulse_interval + pulse_width

    return x, y

# generate pulse sequence
time_points, pulse_sequence = generate_pulse_sequence()

# save pulse sequence
data = torch.Tensor(pulse_sequence).unsqueeze(-1).unsqueeze(-1).numpy()
np.save('datasets/raw_data/Pulse/Pulse.npy', data)
