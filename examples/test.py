import gs
import torch

m = gs.Matrix(torch.classes.gs_classes.Graph(True))

@torch.jit.script
def sampling(A: gs.Matrix):
    t = torch.ones([2], dtype=torch.long)
    return A[:, t]
    # return A.columnwise_slicing(t)
    
# traced = torch.jit.trace(sampling, (m._graph,))
# print(traced.graph)
print(sampling.graph)

