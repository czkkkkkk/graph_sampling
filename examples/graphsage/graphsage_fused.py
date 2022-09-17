from gs.jit.passes import dce
from typing import List
import gs
import torch
import examples.load_graph as load_graph
import time
import numpy as np


def graphsage(A: gs.Matrix, seeds: torch.Tensor, fanouts: List):
    torch.cuda.nvtx.range_push('matrix sampler')
    input_node = seeds
    ret = []
    for fanout in fanouts:
        torch.cuda.nvtx.range_push('colwise slicing')
        subA = A[:, seeds]
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push('colwise sampling')
        subA = subA.columnwise_sampling(fanout, True)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push('all indices')
        seeds = subA.all_indices()
        torch.cuda.nvtx.range_pop()
        ret.append(subA)  # [todo] maybe bug before subA.row_indices
    output_node = seeds
    torch.cuda.nvtx.range_pop()
    return input_node, output_node, ret


dataset = load_graph.load_reddit()
dgl_graph = dataset[0]
m = gs.Matrix(gs.Graph(False))
m.load_dgl_graph(dgl_graph)
print("Check load successfully:", m._graph._CAPI_metadata(), '\n')
seeds = torch.arange(0, 5000).long().cuda()

compiled_func = gs.jit.compile(func=graphsage, args=(m, seeds, [25, 15]))


def slicing_and_sampling_fuse(gm):
    """
    Fuses columnwise_slicing and columnwise_sampling
    """
    for node in gm.graph.nodes:
        if node.target == 'columnwise_sampling' and node.args[
                0].target == 'columnwise_slicing':
            if len(node.args[0].users) > 1:
                continue
            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_method(
                    'fused_columnwise_slicing_sampling',
                    args=(
                        *node.args[0].args,
                        *node.args[1:],
                    ))
                node.replace_all_uses_with(new_node)
    gm.graph.lint()
    gm.recompile()
    return gm


compiled_func.gm = dce(slicing_and_sampling_fuse(compiled_func.gm))


def bench(func, args):
    time_list = []
    for i in range(100):
        # print(i)
        torch.cuda.synchronize()
        begin = time.time()

        ret = func(*args)

        torch.cuda.synchronize()
        end = time.time()

        time_list.append(end - begin)

    print("fused graphsage sampling AVG:",
          np.mean(time_list[10:]) * 1000, " ms.")


bench(graphsage, args=(
    m,
    seeds,
    [25, 15],
))
