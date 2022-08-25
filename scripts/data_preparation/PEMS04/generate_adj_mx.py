import os
import csv
import pickle

import numpy as np


def get_adjacency_matrix(distance_df_filename: str, num_of_vertices: int, id_filename: str = None) -> tuple:
    """Generate adjacency matrix.

    Args:
        distance_df_filename (str): path of the csv file contains edges information
        num_of_vertices (int): number of vertices
        id_filename (str, optional): id filename. Defaults to None.

    Returns:
        tuple: two adjacency matrix.
            np.array: connectivity-based adjacency matrix A (A[i, j]=0 or A[i, j]=1)
            np.array: distance-based adjacency matrix A
    """

    if "npy" in distance_df_filename:
        adj_mx = np.load(distance_df_filename)
        return adj_mx, None
    else:
        adjacency_matrix_connectivity = np.zeros((int(num_of_vertices), int(
            num_of_vertices)), dtype=np.float32)
        adjacency_matrix_distance = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                                             dtype=np.float32)
        if id_filename:
            # the id in the distance file does not start from 0, so it needs to be remapped
            with open(id_filename, "r") as f:
                id_dict = {int(i): idx for idx, i in enumerate(
                    f.read().strip().split("\n"))}  # map node idx to 0-based index (start from 0)
            with open(distance_df_filename, "r") as f:
                f.readline()  # omit the first line
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    adjacency_matrix_connectivity[id_dict[i], id_dict[j]] = 1
                    adjacency_matrix_connectivity[id_dict[j], id_dict[i]] = 1
                    adjacency_matrix_distance[id_dict[i],
                                              id_dict[j]] = distance
                    adjacency_matrix_distance[id_dict[j],
                                              id_dict[i]] = distance
            return adjacency_matrix_connectivity, adjacency_matrix_distance
        else:
            # ids in distance file start from 0
            with open(distance_df_filename, "r") as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    adjacency_matrix_connectivity[i, j] = 1
                    adjacency_matrix_connectivity[j, i] = 1
                    adjacency_matrix_distance[i, j] = distance
                    adjacency_matrix_distance[j, i] = distance
            return adjacency_matrix_connectivity, adjacency_matrix_distance


def generate_adj_pems04():
    distance_df_filename, num_of_vertices = "datasets/raw_data/PEMS04/PEMS04.csv", 307
    if os.path.exists(distance_df_filename.split(".", maxsplit=1)[0] + ".txt"):
        id_filename = distance_df_filename.split(".", maxsplit=1)[0] + ".txt"
    else:
        id_filename = None
    adj_mx, distance_mx = get_adjacency_matrix(
        distance_df_filename, num_of_vertices, id_filename=id_filename)
    # the self loop is missing
    add_self_loop = False
    if add_self_loop:
        print("adding self loop to adjacency matrices.")
        adj_mx = adj_mx + np.identity(adj_mx.shape[0])
        distance_mx = distance_mx + np.identity(distance_mx.shape[0])
    else:
        print("kindly note that there is no self loop in adjacency matrices.")
    with open("datasets/raw_data/PEMS04/adj_PEMS04.pkl", "wb") as f:
        pickle.dump(adj_mx, f)
    with open("datasets/raw_data/PEMS04/adj_PEMS04_distance.pkl", "wb") as f:
        pickle.dump(distance_mx, f)
