import numpy as np
import pickle as pkl
name  = 'METR-LA'
name  = 'PEMS-BAY'

data  = pkl.load(open('datasets/{0}/data.pkl'.format(name), 'rb'))        # 34272, 207, 3
args  = pkl.load(open('datasets/{0}/args.pkl'.format(name), 'rb'))
index = pkl.load(open('datasets/{0}/index.pkl'.format(name), 'rb'))
