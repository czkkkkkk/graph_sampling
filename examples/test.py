from gs import Graph
import torch

A = Graph(False)
indptr = torch.LongTensor([0, 1, 1, 3, 4]).to('cuda:0')
indices = torch.LongTensor([4, 0, 1, 2]).to('cuda:0')
column_ids = torch.LongTensor([2, 3]).to('cuda:0')
A.load_csc(indptr, indices)
subA = A.columnwise_slicing(column_ids)
subA.print()

gm = gs_symbolic_trace(wrapper)
print(gm.graph)
t = torch.ones(10)
for i, j in zip(gm(t), wrapper(t)):
    assert (i._graph.get().equal(j._graph.get()))
