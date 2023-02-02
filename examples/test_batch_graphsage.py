import numpy as np
import gs
import torch
from gs.utils import create_block_from_csc
from gs.utils import SeedGenerator, load_reddit
import numpy as np
import time
from tqdm import tqdm

torch.manual_seed(1)

g, features, labels, n_classes, splitted_idx = load_reddit()
g = g.long().to('cuda')
train_nid = splitted_idx['train'].cuda()
val_nid = splitted_idx['valid'].cuda()
nid = torch.cat([train_nid, val_nid])
indptr, indices, _ = g.adj_sparse('csc')

batch_size = 65536
small_batch_size = 256
num_batchs = 256
fanouts = [15, 10]

A = gs.Graph(False)
A._CAPI_load_csc(indptr, indices)
orig_seeds_ptr = torch.arange(
    num_batchs + 1, dtype=torch.int64, device='cuda') * small_batch_size

# graphsage (batch)
time_list = []
seedloader = SeedGenerator(nid, batch_size=batch_size,
                           shuffle=False, drop_last=False)
for epoch in range(5):
    begin = time.time()
    for it, seeds in enumerate(tqdm(seedloader)):
        seeds_ptr = orig_seeds_ptr
        for fanout in fanouts:
            subA = A._CAPI_fused_columnwise_slicing_sampling(
                seeds, fanout, False)
            indptr, indices, indices_ptr = subA.GetBatchCSC(seeds_ptr)
            unique_tensor, unique_tensor_ptr, [seeds, indices], [
                seeds_ptr, indices_ptr
            ] = torch.ops.gs_ops.BatchRelabel([seeds, indices],
                                              [seeds_ptr, indices_ptr], num_batchs)

            unit = torch.ops.gs_ops.SplitByOffset(unique_tensor,
                                                  unique_tensor_ptr.cpu())
            ptrt = torch.ops.gs_ops.IndptrSplitByOffset(indptr, seeds_ptr)
            indt = torch.ops.gs_ops.SplitByOffset(indices, indices_ptr.cpu())

            for unique_tensor, indptr_tensor, indices_tensor in zip(unit, ptrt, indt):
                block = create_block_from_csc(indptr_tensor,
                                              indices_tensor,
                                              torch.tensor([]),
                                              num_src=unique_tensor.numel(),
                                              num_dst=indptr_tensor.numel() - 1)
                block.srcdata['_ID'] = unique_tensor
            seeds, seeds_ptr = unique_tensor, unique_tensor_ptr
    torch.cuda.synchronize()
    end = time.time()
    time_list.append(end - begin)

print("w/ batching:", np.mean(time_list[2:]))

time_list = []
seedloader = SeedGenerator(nid, batch_size=batch_size,
                           shuffle=False, drop_last=False)
for epoch in range(5):
    begin = time.time()
    for it, seeds in enumerate(tqdm(seedloader)):
        for batch_seeds in torch.split(seeds, small_batch_size):
            for fanout in fanouts:
                subA = A._CAPI_fused_columnwise_slicing_sampling(
                    batch_seeds, fanout, False)
                unique_tensor, num_row, num_col, format_tensor1, format_tensor2, e_ids, format = subA._CAPI_relabel()
                block = create_block_from_csc(format_tensor1,
                                              format_tensor2,
                                              torch.tensor([]),
                                              num_src=num_row,
                                              num_dst=num_col)
                block.srcdata['_ID'] = unique_tensor
                batch_seeds = unique_tensor
    torch.cuda.synchronize()
    end = time.time()
    time_list.append(end - begin)

print("w/o batching:", np.mean(time_list[2:]))
