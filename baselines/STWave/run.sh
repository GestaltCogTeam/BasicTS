#!/bin/bash
python experiments/train.py -c baselines/STWave/METR-LA.py --gpus '0'
python experiments/train.py -c baselines/STWave/PEMS-BAY.py --gpus '0'
python experiments/train.py -c baselines/STWave/PEMS03.py --gpus '0'
python experiments/train.py -c baselines/STWave/PEMS04.py --gpus '0'
python experiments/train.py -c baselines/STWave/PEMS07.py --gpus '0'
python experiments/train.py -c baselines/STWave/PEMS08.py --gpus '0'
