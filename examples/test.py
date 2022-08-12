from gs import Graph, Matrix
import gs
import torch

A = Graph(False)
indptr = torch.LongTensor([0, 1, 1, 3, 4]).to('cuda:0')
indices = torch.LongTensor([4, 0, 1, 2]).to('cuda:0')
column_ids = torch.LongTensor([2, 3]).to('cuda:0')
A.load_csc(indptr, indices)
subA = A.columnwise_slicing(column_ids)
subA.print()


def sampling(A: Matrix):
    column_ids = torch.LongTensor([2, 3]).to('cuda:0')
    return A[:, column_ids]


m = Matrix(A)
compiled_func = gs.jit.compile(func=sampling, args=(m, ))
print("\nOrigin:")
sampling(m)._graph.print()

print()

print("\nCompiled:")
print(compiled_func(m, ))
compiled_func(m, )._graph.print()
