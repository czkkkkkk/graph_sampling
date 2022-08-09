import gs
import torch


def sampling(A: gs.Matrix, t: torch.Tensor):
    return A[:, t * 2], A[t * 1, :], A[t * 1, t * 2], A[:, :]


_graph = torch.classes.gs_classes.Graph(torch.ones(10))
m = gs.Matrix(_graph)
t = torch.ones(10)

compiled_func = gs.jit.compile_class(sampling, (m, t))

for i in compiled_func(m, t):
    print(i._graph.get())

for i in compiled_func(m, torch.ones(10) * 20):
    print(i._graph.get())

for i in compiled_func(
        gs.Matrix(torch.classes.gs_classes.Graph(torch.ones(10) * 10)), t):
    print(i._graph.get())