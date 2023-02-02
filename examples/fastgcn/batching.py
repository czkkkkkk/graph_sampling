from dgl.data import RedditDataset
from gs.utils import create_block_from_csc
import dgl
import gs
from gs.jit.passes import dce
from gs.utils import SeedGenerator, ConvModel, load_reddit
import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
import numpy as np
import time
import argparse


device = torch.device('cuda')
time_list = []
batch_size = 65536
small_batch_size = 256
num_batchs = 256

torch.manual_seed(1)

g = RedditDataset(True)[0].long().to('cuda')
probs = g.out_degrees().float().cuda()
indptr, indices, _ = g.adj_sparse('csc')

batch_size = 65536
small_batch_size = 256
num_batchs = 256
seeds = torch.arange(g.num_nodes()).long().cuda()
idx = torch.randperm(seeds.numel())
original_seeds = seeds[idx[0:batch_size]]
print(original_seeds)
fanouts = [500, 500]

graph = gs.Graph(False)
graph._CAPI_load_csc(indptr, indices)

# graphsage (batch)
time_list = []
for _ in range(20):
    begin = time.time()
    for fanout in fanouts:
        subg = graph._CAPI_slicing(seeds, 0, gs._CSC, gs._CSC, False)
        seeds, seeds_ptr, indptr, indices, indices_ptr = subg.GetBatchCSC(
            small_batch_size)
        neighbors, neighbors_ptr = torch.ops.gs_ops.BatchUnique(
            [indices], [indices_ptr], num_batchs)
        node_probs = probs[neighbors]
        selected, _, selected_ptr = torch.ops.gs_ops.batch_list_sampling_with_probs(
            neighbors, node_probs, fanout, False, neighbors_ptr)
        exit()
        subg = subg._CAPI_batch_slicing(
            selected, 1, gs._CSC, gs._CSC, selected_ptr)
        seeds, seeds_ptr, indptr, indices, indices_ptr = subg._CAPI_GetBatchCSC(
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
            # block = create_block_from_csc(indptr_tensor,
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
        for fanout in fanouts:
            subg = graph._CAPI_slicing(seeds, 0, gs._CSC, gs._CSC)
            neighbors = subg._CAPI_get_valid_rows()
            node_probs = probs[neighbors]
            selected, _ = torch.ops.gs_ops.batch_list_sampling_with_probs(
                neighbors, node_probs, fanout, False, neighbors_ptr)
            subg = subg._CAPI_batch_slicing(selected, 1, gs._CSC, gs._CSC)
            unique_tensor, num_row, num_col, format_tensor1, format_tensor2, e_ids, format = subg._CAPI_relabel(
            )
            # block = create_block_from_csc(format_tensor1,
            #                              format_tensor2,
            #                              torch.tensor([]),
            #                              num_src=num_row,
            #                              num_dst=num_col)
            #block.srcdata['_ID'] = unique_tensor
    torch.cuda.synchronize()
    end = time.time()
    time_list.append(end - begin)

print("w/o batching:", np.mean(time_list[10:]) * 1000)
