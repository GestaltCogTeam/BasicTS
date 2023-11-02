#!/bin/bash
python experiments/train.py -c baselines/MRIformer/ETTh1.py --gpus '0'
python experiments/train.py -c baselines/MRIformer/ETTh2.py --gpus '0'
python experiments/train.py -c baselines/MRIformer/ETTm1.py --gpus '0'
python experiments/train.py -c baselines/MRIformer/ETTm2.py --gpus '0'
python experiments/train.py -c baselines/MRIformer/Electricity.py --gpus '0'
python experiments/train.py -c baselines/MRIformer/ExchangeRate.py --gpus '0'
python experiments/train.py -c baselines/MRIformer/Weather.py --gpus '0'
