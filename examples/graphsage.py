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
compiled_func = gs.jit.compile(func=graphsage, args=(m, seeds, [25, 15]))
print(compiled_func.gm)
input_node, output_node, matrixs = compiled_func(m, seeds, [25, 15])
print("ret input_node:", input_node.numel(), input_node, '\n')
print("ret output_node:", output_node.numel(), output_node, '\n')
for g in matrixs:
    print(g._graph._CAPI_metadata())

from gs.jit.passes import dce


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
print(compiled_func.gm)

input_node, output_node, matrixs = compiled_func(m, seeds, [25, 15])
print("ret input_node:", input_node.numel(), input_node, '\n')
print("ret output_node:", output_node.numel(), output_node, '\n')
for g in matrixs:
    print(g._graph._CAPI_metadata())