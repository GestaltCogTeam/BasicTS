import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from basicts.archs.Stat_arch import SimpleMovingAverage, AutoRegressive, VectorAutoRegression
from basicts.archs.MTGNN_arch import MTGNN
sma  = SimpleMovingAverage(12, 12, 12)
data = torch.randn(64, 12, 207, 3)
pred = sma(data)

wma  = AutoRegressive(12, 12, 12)
data = torch.randn(64, 12, 207, 3)
pred = wma(data)

var  = VectorAutoRegression(12, 12, 12, 207)
data = torch.randn(64, 12, 207, 3)
pred = var(data)
a = 1