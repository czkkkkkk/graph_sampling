from gs import Graph, Matrix
import gs
import torch

A = Graph(False)
indptr = torch.LongTensor([0, 1, 1, 3, 4]).to('cuda:0')
indices = torch.LongTensor([4, 0, 1, 2]).to('cuda:0')
column_ids = torch.LongTensor([2, 3]).to('cuda:0')
A._CAPI_load_csc(indptr, indices)
subA = A._CAPI_columnwise_slicing(column_ids)
print(subA._CAPI_metadata())

print("subA all_indices:", subA._CAPI_all_indices(True))

print("subA relabel result:")
subA_rebaled = subA._CAPI_relabel()
for i in subA_rebaled:
    print(i)
