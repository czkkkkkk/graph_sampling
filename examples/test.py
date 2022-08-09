from difflib import Match
import gs
import torch
from gs import gs_symbolic_trace


def sampling(A: gs.Matrix, t: torch.Tensor):
    return A[:, t * 2], A[t * 1, :], A[t * 1, t * 2], A[:, :]


_graph = torch.classes.gs_classes.Graph(torch.ones(10))


def wrapper(data):
    m = gs.Matrix(_graph)
    return sampling(m, data)


gm = gs_symbolic_trace(wrapper)
print(gm.graph)
t = torch.ones(10)
for i, j in zip(gm(t), wrapper(t)):
    assert (i._graph.get().equal(j._graph.get()))
