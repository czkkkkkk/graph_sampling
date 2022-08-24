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

print("subA all_indices:", subA.all_indices())

print("subA relabel result:")
subA_rebaled = subA.relabel()
print("subA relabel frontier:", subA_rebaled[0])
print("subA relbael indptr:", subA_rebaled[1])
print("subA rebel indices:", subA_rebaled[2])
