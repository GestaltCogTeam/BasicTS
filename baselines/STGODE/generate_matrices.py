# STGODE has a different way of generating the matrices, so we need to use this script to generate the matrices for STGODE
import os
import sys
# TODO: remove it when basicts can be installed by pip
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import csv
import pickle
import argparse

import torch
import numpy as np
from tqdm import tqdm
from fastdtw import fastdtw

from basicts.utils.serialization import load_pkl


def get_normalized_adj(A):
    """
    Returns a tensor, the degree normalized adjacency matrix.
    """
    alpha = 0.8
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)
    return torch.from_numpy(A_reg.astype(np.float32))


def generate_dtw_spa_matrix(dataset_name, in_len, out_len, sigma1=0.1, thres1=0.6, sigma2=10, thres2=0.5, re_scale=True):
    """read data, generate spatial adjacency matrix and semantic adjacency matrix by dtw

    Args:
        sigma1: float, default=0.1, sigma for the semantic matrix
        sigma2: float, default=10, sigma for the spatial matrix
        thres1: float, default=0.6, the threshold for the semantic matrix
        thres2: float, default=0.5, the threshold for the spatial matrix

    Returns:
        data: tensor, T * N * 1
        dtw_matrix: array, semantic adjacency matrix
        sp_matrix: array, spatial adjacency matrix
    """

    # original STGODE use the full time series to generate the matrices, which is not reasonable since the test set is not available in real world
    data_file = "./datasets/{0}/data_in_{1}_out_{2}_rescale_{3}.pkl".format(dataset_name, in_len, out_len, re_scale)
    with open(data_file, 'rb') as f:
        data = pickle.load(f)["processed_data"]
    num_node = data.shape[1]
    if not os.path.exists('{0}/{1}_dtw_distance.npy'.format(os.path.abspath(__file__ + "/.."), dataset_name)):
        print("generate dtw distance matrix")
        data_mean = np.mean([data[:, :, 0][24*12*i: 24*12*(i+1)] for i in range(data.shape[0]//(24*12))], axis=0)
        data_mean = data_mean.squeeze().T 
        dtw_distance = np.zeros((num_node, num_node))
        for i in tqdm(range(num_node)):
            for j in range(i, num_node):
                dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
        for i in range(num_node):
            for j in range(i):
                dtw_distance[i][j] = dtw_distance[j][i]
        np.save('{0}/{1}_dtw_distance.npy'.format(os.path.abspath(__file__ + "/.."), dataset_name), dtw_distance)

    dist_matrix = np.load('{0}/{1}_dtw_distance.npy'.format(os.path.abspath(__file__ + "/.."), dataset_name))

    mean = np.mean(dist_matrix)
    std = np.std(dist_matrix)
    dist_matrix = (dist_matrix - mean) / std
    sigma = sigma1
    dist_matrix = np.exp(-dist_matrix ** 2 / sigma ** 2)
    dtw_matrix = np.zeros_like(dist_matrix)
    dtw_matrix[dist_matrix > thres1] = 1

    # STGODE provides the scripts to generate spatial matrix for PEMS03, PEMS04, PEMS07, PEMS08
    # For other datasets, we use the original spatial matrix.    
    if dataset_name in ["PEMS03", "PEMS04", "PEMS07", "PEMS08"]:
        if not os.path.exists('{0}/{1}_spatial_distance.npy'.format(os.path.abspath(__file__ + "/.."), dataset_name)):
            graph_csv_file_path = "./datasets/raw_data/{0}/{0}.csv".format(dataset_name)
            with open(graph_csv_file_path, 'r') as fp:
                dist_matrix = np.zeros((num_node, num_node)) + np.float('inf')
                file = csv.reader(fp)
                for line in file:
                    break
                for line in file:
                    start = int(line[0])
                    end = int(line[1])
                    dist_matrix[start][end] = float(line[2])
                    dist_matrix[end][start] = float(line[2])
                np.save('{0}/{1}_spatial_distance.npy'.format(os.path.abspath(__file__ + "/.."), dataset_name), dist_matrix)

        dist_matrix = np.load('{0}/{1}_spatial_distance.npy'.format(os.path.abspath(__file__ + "/.."), dataset_name))
        # normalization
        std = np.std(dist_matrix[dist_matrix != np.float('inf')])
        mean = np.mean(dist_matrix[dist_matrix != np.float('inf')])
        dist_matrix = (dist_matrix - mean) / std
        sigma = sigma2
        sp_matrix = np.exp(- dist_matrix**2 / sigma**2)
        sp_matrix[sp_matrix < thres2] = 0 
    else:
        spatial_distance_file = "./datasets/{0}/adj_mx.pkl".format(dataset_name)
        sp_matrix = load_pkl(spatial_distance_file)[-1]

    print(f'average degree of spatial graph is {np.sum(sp_matrix > 0)/2/num_node}')
    print(f'average degree of semantic graph is {np.sum(dtw_matrix > 0)/2/num_node}')
    # normalize
    dtw_matrix = get_normalized_adj(dtw_matrix)
    sp_matrix = get_normalized_adj(sp_matrix)
    return dtw_matrix, sp_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    generate_dtw_spa_matrix("PEMS04", 12, 12, re_scale=True)
