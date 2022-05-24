"""
Code ref: https://github.com/zhengchuanpan/GMAN/blob/master/METR/node2vec/generateSE.py
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import argparse
import node2vec
import networkx as nx
from gensim.models import Word2Vec
from basicts.utils.serialization import load_pkl

def generate_node_embeddings(args):
    try:
        # METR and PEMS_BAY
        _, _, adj_mx = load_pkl("datasets/{0}/adj_mx.pkl".format(args.dataset_name))
    except:
        # PEMS0X
        adj_mx = load_pkl("datasets/{0}/adj_mx.pkl".format(args.dataset_name))

    nx_G = nx.from_numpy_array(adj_mx, create_using=nx.DiGraph())
    G = node2vec.Graph(nx_G, args.is_directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)

    walks = [list(map(str, walk)) for walk in walks]

    model = Word2Vec(walks, vector_size = args.vector_size, window = 10, min_count=0, sg=1, workers = 8, epochs = args.epochs)
    model.wv.save_word2vec_format("datasets/{0}/node2vec_emb.txt".format(args.dataset_name))

if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='METR-LA', help='dataset name.')
    parser.add_argument("--is_directed", type=bool, default=True, help="direct graph.")
    parser.add_argument("--p", type=int, default=2, help="p in node2vec.",)
    parser.add_argument("--q", type=int, default=1, help="q in node2vec.",)
    parser.add_argument("--num_walks", type=int, default=100, help="number of walks..",)
    parser.add_argument("--vector_size", type=int, default=64, help='dimension of node vector.')
    parser.add_argument("--walk_length", type=int, default=80, help='walk length.')
    parser.add_argument("--epochs", type=int, default=1000, help='epochs')
    args    = parser.parse_args()
    print(args)
    generate_node_embeddings(args)
