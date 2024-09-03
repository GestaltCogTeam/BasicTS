import json
import pickle
import numpy as np
from .adjacent_matrix_norm import (
    calculate_scaled_laplacian,
    calculate_symmetric_normalized_laplacian,
    calculate_symmetric_message_passing_adj,
    calculate_transition_matrix
)


def get_regular_settings(dataset_name: str) -> dict:
    """
    Get the regular settings for a dataset.
    
    Args:
        dataset_name (str): Name of the dataset.
    
    Returns:
        dict: Regular settings for the dataset.
    """

    # read json file: datasets/dataset_name/desc.json
    desc = load_dataset_desc(dataset_name)
    regular_settings = desc['regular_settings']
    return regular_settings

def load_dataset_desc(dataset_name: str) -> str:
    """
    Get the description of a dataset.
    
    Args:
        dataset_name (str): Name of the dataset.
    
    Returns:
        str: Description of the dataset.
    """

    # read json file: datasets/dataset_name/desc.json
    with open(f'datasets/{dataset_name}/desc.json', 'r') as f:
        desc = json.load(f)
    return desc

def load_dataset_data(dataset_name: str) -> np.ndarray:
    """
    Load data from a .dat file (memmap) via numpy.

    Args:
        dataset_name (str): Path to the .dat file.

    Returns:
        np.ndarray: Loaded data.
    """

    shape = load_dataset_desc(dataset_name)['shape']
    dat_file_path = f'datasets/{dataset_name}/data.dat'
    data = np.memmap(dat_file_path, mode='r', dtype=np.float32, shape=tuple(shape)).copy()
    return data

def load_pkl(pickle_file: str) -> object:
    """
    Load data from a pickle file.

    Args:
        pickle_file (str): Path to the pickle file.

    Returns:
        object: Loaded object from the pickle file.
    """

    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f'Unable to load data from {pickle_file}: {e}')
        raise
    return pickle_data

def dump_pkl(obj: object, file_path: str):
    """
    Save an object to a pickle file.

    Args:
        obj (object): Object to save.
        file_path (str): Path to the file.
    """

    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def load_adj(file_path: str, adj_type: str):
    """
    Load and preprocess an adjacency matrix.

    Args:
        file_path (str): Path to the file containing the adjacency matrix.
        adj_type (str): Type of adjacency matrix preprocessing.

    Returns:
        list: List of processed adjacency matrices.
        np.ndarray: Raw adjacency matrix.
    """

    try:
        _, _, adj_mx = load_pkl(file_path)
    except ValueError:
        adj_mx = load_pkl(file_path)

    if adj_type == 'scalap':
        adj = [calculate_scaled_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == 'normlap':
        adj = [calculate_symmetric_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == 'symnadj':
        adj = [calculate_symmetric_message_passing_adj(adj_mx).astype(np.float32).todense()]
    elif adj_type == 'transition':
        adj = [calculate_transition_matrix(adj_mx).T]
    elif adj_type == 'doubletransition':
        adj = [calculate_transition_matrix(adj_mx).T, calculate_transition_matrix(adj_mx.T).T]
    elif adj_type == 'identity':
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    elif adj_type == 'original':
        adj = [adj_mx]
    else:
        raise ValueError('Undefined adjacency matrix type.')

    return adj, adj_mx
