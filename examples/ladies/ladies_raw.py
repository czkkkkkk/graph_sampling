import gs
import torch
from gs.utils import load_reddit
import time
import numpy as np


def ladies(A: gs.Matrix, seeds: torch.Tensor, fanouts: list):
    input_node = seeds
    ret = []
    D_in = A.sum(axis=0)
    D_out = A.sum(axis=1)
    P = A.divide(D_out.sqrt(), axis=1).divide(D_in.sqrt(), axis=0)
    for fanout in fanouts:
        U = P[:, seeds]
        prob = U.l2norm(axis=1)
        selected, _ = torch.ops.gs_ops.list_sampling_with_probs(
            U.row_indices(unique=False), prob + 1, fanout, False)
        nodes = torch.cat((seeds, selected)).unique()  # add self-loop
        subU = U[nodes, :]
        subU = subU.divide(prob[nodes], axis=1).normalize(axis=1)
        seeds = subU.all_indices(unique=True)
        ret.insert(0, subU.to_dgl_block())
    output_node = seeds
    return input_node, output_node, ret


dataset = load_reddit()
dgl_graph = dataset[0]
m = gs.Matrix(gs.Graph(False))
m.load_dgl_graph(dgl_graph)
print("Check load successfully:", m._graph._CAPI_metadata(), '\n')
seeds = torch.arange(0, 1000).long().cuda()

# compiled_func = gs.jit.compile(func=ladies, args=(m, seeds, [2000, 2000]))


def bench(func, args):
    time_list = []
    for i in range(100):
        torch.cuda.synchronize()
        begin = time.time()

        ret = func(*args)

        torch.cuda.synchronize()
        end = time.time()

        time_list.append(end - begin)

    print("ladies sampling AVG:", np.mean(time_list[10:]) * 1000, " ms.")


bench(ladies, args=(m, seeds, [2000, 2000]))
