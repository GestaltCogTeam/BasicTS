#!/bin/bash
python experiments/train.py -c baselines/STAEformer/METR-LA.py --gpus '0'
python experiments/train.py -c baselines/STAEformer/PEMS-BAY.py --gpus '0'
python experiments/train.py -c baselines/STAEformer/PEMS03.py --gpus '0'
python experiments/train.py -c baselines/STAEformer/PEMS04.py --gpus '0'
python experiments/train.py -c baselines/STAEformer/PEMS07.py --gpus '0'
python experiments/train.py -c baselines/STAEformer/PEMS08.py --gpus '0'
