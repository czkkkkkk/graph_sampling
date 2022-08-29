from typing import List
import gs
import torch
import load_graph
import time
import numpy as np


time_list = []


def graphsage(A: gs.Matrix, seeds: torch.Tensor, fanouts: List):
    input_node = seeds
    ret = []
    acc_time = 0
    for fanout in fanouts:
        torch.cuda.synchronize()
        begin = time.time()
        subA = A[:, seeds]
        subA = subA.columnwise_sampling(fanout, True)
        torch.cuda.synchronize()
        end = time.time()
        acc_time += end - begin
        seeds = subA.all_indices()
        ret.append(subA)  # [todo] maybe bug before subA.row_indices
    output_node = seeds
    time_list.append(acc_time)
    return input_node, output_node, ret


# dataset = load_graph.load_reddit()
dataset = load_graph.load_custom_reddit(
    "/home/ubuntu/NextDoorEval/NextDoor/input/reddit.data")
dgl_graph = dataset[0]
m = gs.Matrix(gs.Graph(False))
m.load_dgl_graph(dgl_graph)
print("Check load successfully:", m._graph._CAPI_metadata(), '\n')
# seeds = torch.arange(0, 5000).long().cuda()
seeds = dgl_graph.nodes().cuda()

# compiled_func = gs.jit.compile(func=graphsage, args=(m, seeds, [25, 15]))


def bench(func, args):
    for i in range(10):
        # torch.cuda.synchronize()
        # begin = time.time()

        ret = func(*args)

        # torch.cuda.synchronize()
        # end = time.time()

        # time_list.append(end - begin)

    print("nonfused graphsage sampling AVG:",
          np.mean(time_list[3:]), " s.")


bench(graphsage, args=(
    m,
    seeds,
    [25, 15],
))
