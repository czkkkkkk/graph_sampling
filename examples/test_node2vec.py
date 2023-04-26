from gs import Graph, HeteroGraph, Matrix, HeteroMatrix
import gs
import torch
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

A1 = Graph(False)
indptr1 = torch.LongTensor([0, 0, 1, 2, 3]).to('cuda:0')
indices1 = torch.LongTensor([0, 1, 2]).to('cuda:0')
matrix = gs.Matrix(gs.Graph(False))
matrix._graph._CAPI_load_csc(indptr1, indices1)

seeds = torch.LongTensor([3, 2]).to('cuda:0')
print("random walk fused:")
nodes = matrix._graph._CAPI_node2vec_random_walk(seeds,2,2,0.5)
print("nodes:",nodes)
print(nodes.reshape((-1, seeds.numel())))
