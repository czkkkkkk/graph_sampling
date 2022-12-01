import gs
import torch

A = gs.Graph(False)
indptr = torch.arange(0, 21).long().to('cuda') * 5
indices = torch.arange(0, 100).long().to('cuda')
A._CAPI_load_csc(indptr, indices)

row_ids = torch.arange(0, 100, 5).long().to('cuda')
for t in A._CAPI_metadata():
    if t is not None:
        print(t.numel(), t)

print()

subA = A._CAPI_rowwise_slicing(row_ids)

#    EXPECT_TRUE(csc_ptr->indptr.equal(torch::arange(21).to(torch::kCUDA) * 1));
#    EXPECT_TRUE(csc_ptr->indices.equal(torch::arange(20).to(torch::kCUDA)));

for t in subA._CAPI_metadata():
    if t is not None:
        print(t.numel(), t)