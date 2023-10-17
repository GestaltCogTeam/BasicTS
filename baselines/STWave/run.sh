#!/bin/bash
python experiments/train.py -c baselines/STWave/STWave_METR-LA.py --gpus '0'
python experiments/train.py -c baselines/STWave/STWave_PEMS-BAY.py --gpus '0'
python experiments/train.py -c baselines/STWave/STWave_PEMS03.py --gpus '0'
python experiments/train.py -c baselines/STWave/STWave_PEMS04.py --gpus '0'
python experiments/train.py -c baselines/STWave/STWave_PEMS07.py --gpus '0'
python experiments/train.py -c baselines/STWave/STWave_PEMS08.py --gpus '0'
