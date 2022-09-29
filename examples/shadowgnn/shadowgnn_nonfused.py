from typing import List
import gs
import torch
import load_graph
import time
import numpy as np

device = torch.device('cuda:%d' % 0)

dataset = load_graph.load_reddit()
dgl_graph = dataset[0]
m = gs.Matrix(gs.Graph(False))
m.load_dgl_graph(dgl_graph)

#


def shadowgnn_baseline(A: gs.Matrix, seeds, fanouts):
    torch.cuda.nvtx.range_push("shadowgnn python")
    output_nodes = seeds
    for fanout in reversed(fanouts):
        subA = A[:, seeds]
        subA = subA.columnwise_sampling(fanout, False)
        seeds = subA.all_indices()
    retA = A[seeds, seeds]
    retA = retA.relabel()
    torch.cuda.nvtx.range_pop()
    return seeds, output_nodes, retA


str_list = []


def bench(loop_num, func, args):
    time_list = []
    for i in range(loop_num):
        torch.cuda.synchronize()
        begin = time.time()
        input_nodes, output_nodes, retA = func(*args)
        #print("returned nodes:", input_nodes.numel())
        # assert torch.equal(input_nodes, retA.all_indices())
        # assert torch.equal(input_nodes[:output_nodes.shape[0]], output_nodes)
        # print(retA._CAPI_metadata())
        torch.cuda.synchronize()
        end = time.time()

        time_list.append(end - begin)
    # str_list.append("%d,%d,%.3f" %
    #                 (seed_num, metalength, np.mean(time_list[10:]) * 1000))
    print("matrix shadowgnn AVG:", np.mean(time_list[10:]) * 1000, " ms.")


# seeds_set = [1000, 10000, 50000, 100000, 200000, 2000000, 10000000]
# metapath_len = [5, 10, 15, 20, 25, 30]


fanouts = [5, 15, 25]
#seeds = torch.randint(0, 232965, (5,), device='cuda')
seeds = torch.arange(0, 5).long().cuda()
bench(
    100,
    shadowgnn_baseline, args=(
        m,
        seeds,
        fanouts))
