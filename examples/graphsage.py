import gs
import torch
import dgl

gs.init()

@torch.jit.trace
def sampling(A, seeds, fanouts):
    ret = []
    for fanout in fanouts:
        subA = A[:, seeds]
        subA = subA.columnwise_sampling(fanout)
        ret.append(subA)
        seeds = subA.row_indices(unique=True)
    return ret

func = gs.optimize(sampling)

A = gs.Matrix().load_dgl_graph(DGLGraph(...))
seeds = torch.Tensor(...)
fanout=[25, 10]

subs = func(A, seeds, fanout)
    
