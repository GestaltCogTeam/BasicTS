#!/bin/bash
python experiments/train.py -c baselines/Linear/Linear_PEMS03.py --gpus '3'
python experiments/train.py -c baselines/NLinear/NLinear_PEMS03.py --gpus '3'
python experiments/train.py -c baselines/DLinear/DLinear_PEMS03.py --gpus '3'
