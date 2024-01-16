import os
import sys
import numpy as np
import matplotlib.pyplot as plt
PROJECT_DIR = os.path.abspath(__file__ + "/../../../..")
os.chdir(PROJECT_DIR)


def generate_pulse_sequence(duration, min_interval, max_interval):
    time_points = np.arange(0, duration, 1)
    pulse_sequence = np.zeros_like(time_points)

    current_time = 0
    while current_time < duration:
        pulse_interval = np.random.uniform(min_interval, max_interval)
        pulse_width = 1
        pulse_sequence[int(current_time):int(current_time + pulse_width)] = 1
        current_time += pulse_interval + pulse_width

    return time_points, pulse_sequence

# hyper parameterts
duration = 20000  # time series length
min_interval = 30  # minimum interval between two pulses
max_interval = 30  # maximum interval between two pulses

# generate pulse sequence
time_points, pulse_sequence = generate_pulse_sequence(duration, min_interval, max_interval)

# save pulse sequence
import torch
data = torch.Tensor(pulse_sequence).unsqueeze(-1).unsqueeze(-1).numpy()
np.save('datasets/raw_data/Pulse/Pulse.npy', data)
