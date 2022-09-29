from typing import List
import gs
import torch
import load_graph
import time
import numpy as np


def graphsaint(A: gs.Matrix, seeds_num, walk_length):
    seeds = torch.randint(
        0, 232965, (seeds_num,), device='cuda')
    torch.cuda.nvtx.range_push("graph saint non-fused random walk")
    paths = A.random_walk(seeds, walk_length)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("graph unique")
    node_ids = paths.view(seeds_num*(walk_length+1))
    out = torch.unique(node_ids, sorted=False)
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("graph induce subgraph")
    induced_subA = A[out, out]
    unique_tensor, csc_indptr, csc_indices = induced_subA._graph.relabel()
    retA = gs.Graph(False)
    retA.load_csc(csc_indptr, csc_indices)
    ret_m = gs.Matrix(retA)
    torch.cuda.nvtx.range_pop()
    return ret_m


dataset = load_graph.load_reddit()
dgl_graph = dataset[0]
m = gs.Matrix(gs.Graph(False))
m.load_dgl_graph(dgl_graph)
print("Check load successfully:", m._graph._CAPI_metadata(), '\n')


def bench(func, args):
    time_list = []
    for i in range(100):
        # print(i)
        torch.cuda.synchronize()
        begin = time.time()

        ret = func(*args)
        # print(ret._graph._CAPI_metadata())
        torch.cuda.synchronize()
        end = time.time()
        time_list.append(end - begin)
    print("fused graphsage sampling AVG:",
          np.mean(time_list[10:]) * 1000, " ms.")


bench(graphsaint, args=(
    m,
    2000,
    4,
))
