from json import load
from typing import List
import gs
import torch
import load_graph


def graphsage(A: gs.Matrix, seeds: torch.Tensor, fanouts: List):
    input_node = seeds
    ret = []
    for fanout in fanouts:
        subA = A[:, seeds]
        subA = subA.columnwise_sampling(fanout, True)
        seeds = subA.all_indices()
        ret.append(subA)  # [todo] maybe bug before subA.row_indices
    output_node = seeds
    return input_node, output_node, ret


dataset = load_graph.load_reddit()
dgl_graph = dataset[0]
m = gs.Matrix(gs.Graph(False))
m.load_dgl_graph(dgl_graph)
print("Check load successfully:", m._graph._CAPI_metadata(), '\n')

seeds = torch.arange(0, 5000).long().cuda()
compiled_func = gs.jit.compile(func=graphsage, args=(m, seeds, [25, 10]))
input_node, output_node, matrixs = compiled_func(m, seeds, [2, 2])
print("ret input_node:", input_node.numel(), input_node, '\n')
print("ret output_node:", output_node.numel(), output_node, '\n')
for m in matrixs:
    print(m._graph._CAPI_metadata())
