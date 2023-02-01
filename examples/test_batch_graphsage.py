import numpy as np
import gs
import torch
import dgl

from dgl.data import RedditDataset

g = RedditDataset(True)[0].long().to('cuda')
indptr, indices, _ = g.adj_sparse('csc')
seeds = torch.arange(15).long().cuda()

A = gs.Graph(False)
A._CAPI_load_csc(indptr, indices)


# graphsage (batch)
subA = A._CAPI_fused_columnwise_slicing_sampling(seeds, 15, False)
seeds, seeds_ptr, indptr, indices, indices_ptr = subA.GetBatchCSC(5)
unique_tensor, unique_tensor_ptr, [seeds, indices], [seeds_ptr, indices_ptr] = torch.ops.gs_ops.BatchRelabel(
    [seeds, indices], [seeds_ptr, indices_ptr], 15 // 5)

unique_tensors = torch.ops.gs_ops.SplitByOffset(
    unique_tensor, unique_tensor_ptr.cpu())
indices_tensors = torch.ops.gs_ops.SplitByOffset(indices, indices_ptr.cpu())
indptr_tensors = torch.ops.gs_ops.IndptrSplitBySize(indptr, 5)

print(indptr_tensors)

print(indices_tensors)
