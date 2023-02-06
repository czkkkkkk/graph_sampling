from tqdm import tqdm
import time
from gs.utils import SeedGenerator, load_reddit
from gs.utils import create_block_from_coo
import torch
import gs
import numpy as np

torch.manual_seed(1)

g, features, labels, n_classes, splitted_idx = load_reddit()
g = g.long().to('cuda')
probs = g.out_degrees().float().cuda()
train_nid = splitted_idx['train'].cuda()
val_nid = splitted_idx['valid'].cuda()
nid = torch.cat([train_nid, val_nid])
indptr, indices, _ = g.adj_sparse('csc')

n_epoch = 5
batch_size = 65536
small_batch_size = 256
num_batchs = int(batch_size / small_batch_size)
fanouts = [25, 15]

A = gs.Graph(False)
A._CAPI_load_csc(indptr, indices)
orig_seeds_ptr = torch.arange(
    num_batchs + 1, dtype=torch.int64, device='cuda') * small_batch_size

# graphsage (batch)
time_list = []
layer_time = [[], []]
seedloader = SeedGenerator(nid, batch_size=batch_size,
                           shuffle=False, drop_last=False)
for epoch in range(n_epoch):
    # torch.cuda.synchronize()
    begin = time.time()
    batch_layer_time_1 = 0
    batch_layer_time_2 = 0
    for it, seeds in enumerate(tqdm(seedloader)):
        # torch.cuda.nvtx.range_push('sampling')
        seeds_ptr = orig_seeds_ptr
        if it == 2:
            num_batchs = int(
                (seeds.numel() + small_batch_size - 1) / small_batch_size)
            seeds_ptr = torch.arange(
                num_batchs + 1, dtype=torch.int64, device='cuda') * small_batch_size
            seeds_ptr[-1] = seeds.numel()
        for layer, fanout in enumerate(fanouts):
            # torch.cuda.synchronize()
            layer_start = time.time()
            subA = A._CAPI_slicing(seeds, 0, gs._CSC, gs._CSC + gs._COO, False)
            indptr, indices, indices_ptr = subA.GetBatchCSC(seeds_ptr)
            neighbors, neighbors_ptr = torch.ops.gs_ops.BatchUnique(
                [indices], [indices_ptr], num_batchs)
            node_probs = probs[neighbors]
            selected, _, selected_ptr = torch.ops.gs_ops.batch_list_sampling_with_probs(
                neighbors, node_probs, fanout, False, neighbors_ptr)
            subA, row, col, coo_ptr = subA._CAPI_batch_slicing(
                selected, 1, gs._COO, gs._COO, indices_ptr, selected_ptr)
            # torch.cuda.nvtx.range_push('batchrelabel')
            unique_tensor, unique_tensor_ptr, [col, row], [
                col_ptr, row_ptr
            ] = torch.ops.gs_ops.BatchRelabel([col, row],
                                              [coo_ptr, coo_ptr], num_batchs)
            # torch.cuda.nvtx.range_pop()

            # torch.cuda.nvtx.range_push('splitbyoffsets')
            unit = torch.ops.gs_ops.SplitByOffset(unique_tensor,
                                                  unique_tensor_ptr)
            colt = torch.ops.gs_ops.SplitByOffset(col, col_ptr)
            rowt = torch.ops.gs_ops.SplitByOffset(row, row_ptr)
            # torch.cuda.nvtx.range_pop()

            for unique, col, row in zip(unit, colt, rowt):
                block = create_block_from_coo(row,
                                              col,
                                              num_src=unique.numel(),
                                              num_dst=seeds.numel())
                block.srcdata['_ID'] = unique
                pass
            seeds, seeds_ptr = unique_tensor, unique_tensor_ptr
            # torch.cuda.synchronize()
            layer_end = time.time()
            if layer == 0:
                batch_layer_time_1 += layer_end - layer_start
            else:
                batch_layer_time_2 += layer_end - layer_start
        # torch.cuda.nvtx.range_pop()
    layer_time[0].append(batch_layer_time_1)
    layer_time[1].append(batch_layer_time_2)
    # torch.cuda.synchronize()
    end = time.time()
    time_list.append(end - begin)

print("w/ batching:", np.mean(time_list[2:]))
print("w/ batching layer1:", np.mean(layer_time[0][2:]))
print("w/ batching layer2:", np.mean(layer_time[1][2:]))

time_list = []
layer_time = [[], []]
seedloader = SeedGenerator(nid, batch_size=small_batch_size,
                           shuffle=False, drop_last=False)
for epoch in range(n_epoch):
    # torch.cuda.synchronize()
    begin = time.time()
    batch_layer_time_1 = 0
    batch_layer_time_2 = 0
    for it, seeds in enumerate(tqdm(seedloader)):
        for layer, fanout in enumerate(fanouts):
            # torch.cuda.synchronize()
            layer_start = time.time()
            subA = A._CAPI_slicing(seeds, 0, gs._CSC, gs._CSC + gs._COO, False)
            neighbors = subA._CAPI_get_valid_rows()
            node_probs = probs[neighbors]
            selected, _ = torch.ops.gs_ops.list_sampling_with_probs(
                neighbors, node_probs, fanout, False)
            subA = subA._CAPI_slicing(selected, 1, gs._COO, gs._COO, False)
            unique_tensor, num_row, num_col, format_tensor1, format_tensor2, e_ids, format = subA._CAPI_relabel()
            block = create_block_from_coo(format_tensor1,
                                          format_tensor2,
                                          num_src=num_row,
                                          num_dst=num_col)
            block.srcdata['_ID'] = unique_tensor
            # torch.cuda.synchronize()
            layer_end = time.time()
            if layer == 0:
                batch_layer_time_1 += layer_end - layer_start
            else:
                batch_layer_time_2 += layer_end - layer_start
    layer_time[0].append(batch_layer_time_1)
    layer_time[1].append(batch_layer_time_2)
    # torch.cuda.synchronize()
    end = time.time()
    time_list.append(end - begin)

print("w/o batching:", np.mean(time_list[2:]))
print("w/o batching layer1:", np.mean(layer_time[0][2:]))
print("w/o batching layer2:", np.mean(layer_time[1][2:]))
