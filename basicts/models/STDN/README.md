如果您想要复现实验结果，需要使用特殊的邻接矩阵。该邻接矩阵是基于原始数据中的距离CSV生成的有向图，与常用的无向图不同（[参考](https://github.com/roarer008/STDN/blob/main/utils.py#L146)）。
我们在可选数据中提供了此邻接矩阵数据（optional_data目录中），将对应数据覆盖到dataset目录中即可。

To reproduce the experimental results, you need to use a ​​special adjacency matrix​​. This adjacency matrix is a ​​directed graph​​ generated from the distance CSV file in the raw data, differing from the commonly used undirected graphs ([ref](https://github.com/roarer008/STDN/blob/main/utils.py#L146)). 
We have provided this adjacency matrix data in the optional resources(optional_data directory). If you want, simply overwrite the corresponding data in the dataset directory.