import gs
import torch


@gs.jit.compile
def sampling(A: gs.Matrix, t: torch.Tensor):
    return A[:, t * 2], A[t * 1, :], A[t * 1, t * 2], A[:, :]


_graph = torch.classes.gs_classes.Graph(torch.ones(10))
m = gs.Matrix(_graph)
t = torch.ones(10)
for i in sampling(m, t):
    print(i._graph.get())
