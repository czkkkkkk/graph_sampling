import numpy as np
import gs
import torch
import dgl
from gs.utils import create_block_from_csc
from dgl.data import RedditDataset
import numpy as np
import time

torch.manual_seed(1)

g = RedditDataset(True)[0].long().to('cuda')
indptr, indices, _ = g.adj_sparse('csc')

batch_size = 65536
small_batch_size = 256
num_batchs = 256
seeds = torch.arange(g.num_nodes()).long().cuda()
idx = torch.randperm(seeds.numel())
original_seeds = seeds[idx[0:batch_size]]
print(original_seeds)

A = gs.Graph(False)
A._CAPI_load_csc(indptr, indices)

# graphsage (batch)
time_list = []
for _ in range(20):
    begin = time.time()
    subA = A._CAPI_fused_columnwise_slicing_sampling(original_seeds, 15, False)
    seeds, seeds_ptr, indptr, indices, indices_ptr = subA.GetBatchCSC(
        small_batch_size)
    unique_tensor, unique_tensor_ptr, [seeds, indices], [
        seeds_ptr, indices_ptr
    ] = torch.ops.gs_ops.BatchRelabel([seeds, indices],
                                      [seeds_ptr, indices_ptr], num_batchs)

    #torch.ops.gs_ops.SplitByOffset(unique_tensor, unique_tensor_ptr.cpu())
    #torch.ops.gs_ops.IndptrSplitBySize(indptr, small_batch_size)
    #torch.ops.gs_ops.SplitByOffset(indices, indices_ptr.cpu())

    for unique_tensor, indptr_tensor, indices_tensor in zip(
            torch.ops.gs_ops.SplitByOffset(unique_tensor,
                                           unique_tensor_ptr.cpu()),
            torch.ops.gs_ops.IndptrSplitBySize(indptr, small_batch_size),
            torch.ops.gs_ops.SplitByOffset(indices, indices_ptr.cpu())):
        #block = create_block_from_csc(indptr_tensor,
        #                              indices_tensor,
        #                              torch.tensor([]),
        #                              num_src=unique_tensor.numel(),
        #                              num_dst=indptr_tensor.numel() - 1)
        #block.srcdata['_ID'] = unique_tensor
        break
    torch.cuda.synchronize()
    end = time.time()
    time_list.append(end - begin)

print("w/ batching:", np.mean(time_list[10:]) * 1000)

for _ in range(20):
    begin = time.time()
    for seeds in torch.split(original_seeds, small_batch_size):
        subA = A._CAPI_fused_columnwise_slicing_sampling(seeds, 15, False)
        unique_tensor, num_row, num_col, format_tensor1, format_tensor2, e_ids, format = subA._CAPI_relabel(
        )
        #block = create_block_from_csc(format_tensor1,
        #                              format_tensor2,
        #                              torch.tensor([]),
        #                              num_src=num_row,
        #                              num_dst=num_col)
        #block.srcdata['_ID'] = unique_tensor
    torch.cuda.synchronize()
    end = time.time()
    time_list.append(end - begin)

print("w/o batching:", np.mean(time_list[10:]) * 1000)