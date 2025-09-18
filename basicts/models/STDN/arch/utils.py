import logging
import numpy as np
import pandas as pd
import os
import pickle
import scipy.sparse as sp
import sys
import torch
import csv
from scipy.sparse import linalg

from sklearn.metrics import mean_squared_error  # MSE
from sklearn.metrics import mean_absolute_error  # MAE
from .metrics import masked_mape_np
import torch.nn.functional as F
import random


# statistic model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def seq2instance(data, num_his, num_pred):
    num_step, dims = data.shape
    num_sample = num_step - num_his - num_pred + 1
    x = torch.zeros(num_sample, num_his, dims)
    y = torch.zeros(num_sample, num_pred, dims)
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred]
    return x, y
def load_data(args):
	data_config = args['Data']
	training_config = args['Training']
	# Traffic
	df = pd.read_hdf(data_config['traffic_file'])
	traffic = torch.from_numpy(df.values)
	# train/val/test
	num_step = df.shape[0]
	train_steps = round(float(data_config['train_ratio']) * num_step)
	test_steps = round(float(data_config['test_ratio']) * num_step)
	val_steps = num_step - train_steps - test_steps
	print('traffic shape', traffic.shape)
	train = traffic[: train_steps]
	val = traffic[train_steps: train_steps + val_steps]
	test = traffic[-test_steps:]
	# X, Y
	num_his = int(training_config['num_his'])
	num_pred = int(training_config['num_pred'])
	trainX, trainY = seq2instance(train, num_his, num_pred)
	valX, valY = seq2instance(val, num_his, num_pred)
	testX, testY = seq2instance(test, num_his, num_pred)
	trainX=trainX.unsqueeze(-1)
	trainY = trainY.unsqueeze(-1)
	valX = valX.unsqueeze(-1)
	valY = valY.unsqueeze(-1)
	testX = testX.unsqueeze(-1)
	testY = testY.unsqueeze(-1)
	# normalization
	mean, std = torch.mean(trainX), torch.std(trainX)
	mean=mean.unsqueeze(0)
	std=std.unsqueeze(0)
	trainX = (trainX - mean) / std
	valX = (valX - mean) / std
	testX = (testX - mean) / std

	# temporal embedding
	time = pd.DatetimeIndex(df.index)
	dayofweek = torch.reshape(torch.tensor(time.weekday), (-1, 1))
	timeofday = (time.hour * 3600 + time.minute * 60 + time.second) \
				// (5 * 60)
	timeofday = torch.reshape(torch.tensor(timeofday), (-1, 1))
	time = torch.cat((dayofweek, timeofday), -1)
	# train/val/test
	train = time[: train_steps]
	val = time[train_steps: train_steps + val_steps]
	test = time[-test_steps:]
	# shape = (num_sample, num_his + num_pred, 2)
	trainTE = seq2instance(train, num_his, num_pred)
	trainTE = torch.cat(trainTE, 1).type(torch.int32)
	valTE = seq2instance(val, num_his, num_pred)
	valTE = torch.cat(valTE, 1).type(torch.int32)
	testTE = seq2instance(test, num_his, num_pred)
	testTE = torch.cat(testTE, 1).type(torch.int32)

	return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
			mean, std)

def mae_rmse_mape(y_pred, y_true):
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    loss = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = masked_mape_np(y_true, y_pred, null_val=0)
    return loss, rmse, mape


def get_normalized_adj(path):
    """
    Returns the degree normalized adjacency matrix. 度归一化邻接矩阵
    """
    A = np.load(path)
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float64))

    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    
    diag = np.reciprocal(np.sqrt(D))

    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    del A, D, diag
    return A_wave


def get_adj_from_csv(path, num_of_vertices, id_filename=None):
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

    distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                        dtype=np.float32)
    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  
        
        with open(path, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                distaneA[id_dict[i], id_dict[j]] = distance
        return A, distaneA
    else:
        with open(path, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row)!=3:
                    continue
                i , j, distance = int(row[0]), int(row[1]), float(row[2])
                A[i,j] = 1
                # A[j,i] = 1
                distaneA[i, j] = distance
        return A, distaneA

def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    isolated_point_num = np.sum(np.where(d, 0, 1))
    print(f"Number of isolated points: {isolated_point_num}")
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian, isolated_point_num

def cal_lape(adj_mx):
    lape_dim = 32
    L, isolated_point_num = calculate_normalized_laplacian(adj_mx)
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    laplacian_pe = EigVec[:, isolated_point_num + 1: lape_dim + isolated_point_num + 1]
    return laplacian_pe

def get_adj_from_npy(path):
    data = np.load(path)
    num_of_vertices = data.shape[0]
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32) 
    for i in range(num_of_vertices):
        for j in range(num_of_vertices):
            if i!=j and data[i,j]!=0:
                A[i,j] = 1
    return A

def add_data(X, number = 2):
    size, times, nodes,feature = X.shape

def get_lpls(A):
    lpls = cal_lape(A.copy())
    lpls = torch.from_numpy(np.array(lpls, dtype='float32')).type(torch.FloatTensor)
    return lpls