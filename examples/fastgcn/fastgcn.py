import gs
import torch
from gs.utils import load_graph
from typing import List


def fastgcn_sampler(A: gs.matrix_api.Matrix, seeds: torch.Tensor, fanouts: List):
    input_node = seeds
    ret = []
    for K in fanouts:
        subA = A[:, seeds]
        sampleA = subA.collective_sampling(K, subA.row_ndata['deg'], False)
        seeds = sampleA.all_nodes()
        ret.append(sampleA.to_dgl_block())
    output_node = seeds
    return input_node, output_node, ret


if __name__ == "__main__":
    torch.manual_seed(1)
    dataset = load_graph.load_reddit()
    dgl_graph = dataset[0]
    csc_indptr, csc_indices, _ = dgl_graph.adj_sparse('csc')

    m = gs.matrix_api.Matrix()
    m.load_graph('CSC', [csc_indptr.cuda(), csc_indices.cuda()])

    m.set_row_data('deg', dgl_graph.out_degrees().float().cuda())

    seeds = torch.randint(0, 10000, (500,)).cuda()

    # compile_func = gs.jit.compile(
    #    func=fastgcn_sampler, args=(m, seeds, [2000, 2000]))
    compile_func = fastgcn_sampler
    for i in compile_func(m, seeds, [2000, 2000]):
        print(i)
