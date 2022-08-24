from typing import List
import gs
import torch

def HeteroRandomWalk(HA: gs.HeteroMatrix, seeds: torch.Tensor, meta_path: List):
    input_node = seeds
    ret = []
    for etype in meta_path:
        A = HA.get_homo_matrix(etype)
        subA = A.columnwise_sampling(1, True)
        seeds = subA.row_indices()
        ret.append(seeds)
    return torch.stack(ret)

def HeteroRandomWalkFused(HA: gs.HeteroMatrix, seeds: torch.Tensor, metapath: List):
    return HA.metapath_random_walk(seeds, seeds, metapath)