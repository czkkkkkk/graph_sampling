import numpy as np
import gs
import torch
from gs.utils import SeedGenerator, load_reddit, load_ogb, create_block_from_csc
import numpy as np
import time
from tqdm import tqdm

torch.manual_seed(1)

g, features, labels, n_classes, splitted_idx = load_ogb(
    'ogbn-products', '/home/ubuntu/gs-experiments/datasets')
g = g.long().to('cuda')
train_nid = splitted_idx['train'].cuda()
val_nid = splitted_idx['valid'].cuda()
nid = torch.cat([train_nid, val_nid])
indptr, indices, _ = g.adj_sparse('csc')

n_epoch = 5
batch_size = 65536
small_batch_size = 256
num_batchs = int((batch_size + small_batch_size - 1) / small_batch_size)
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
    torch.cuda.synchronize()
    begin = time.time()
    batch_layer_time_1 = 0
    batch_layer_time_2 = 0
    for it, seeds in enumerate(tqdm(seedloader)):
        # torch.cuda.nvtx.range_push('sampling')
        num_batchs = int(
            (batch_size + small_batch_size - 1) / small_batch_size)
        seeds_ptr = orig_seeds_ptr
        if it == len(seedloader) - 1:
            num_batchs = int(
                (seeds.numel() + small_batch_size - 1) / small_batch_size)
            seeds_ptr = torch.arange(
                num_batchs + 1, dtype=torch.int64, device='cuda') * small_batch_size
            seeds_ptr[-1] = seeds.numel()
        for layer, fanout in enumerate(fanouts):
            print(seeds.numel())
            torch.cuda.synchronize()
            layer_start = time.time()
            subA = A._CAPI_fused_columnwise_slicing_sampling(
                seeds, fanout, False)
            # torch.cuda.nvtx.range_push('getbatchcsc')
            indptr, indices, indices_ptr = subA.GetBatchCSC(seeds_ptr)
            # torch.cuda.nvtx.range_pop()
            # torch.cuda.nvtx.range_push('batchrelabel')
            data, data_key, data_ptr = torch.ops.gs_ops.BatchConcat(
                [seeds, indices], [seeds_ptr, indices_ptr])
            unique_tensor, unique_tensor_ptr, relabel_data, relabel_data_ptr = torch.ops.gs_ops.BatchRelabelByKey(
                data, data_ptr, data_key, data, data_ptr, data_key)
            torch.ops.gs_ops.BatchSplit(relabel_data, relabel_data_ptr, data_key,
                                        [seeds, indices], [seeds_ptr, indices_ptr])
            # torch.cuda.nvtx.range_pop()

            # torch.cuda.nvtx.range_push('splitbyoffsets')
            unit = torch.ops.gs_ops.SplitByOffset(unique_tensor,
                                                  unique_tensor_ptr)
            ptrt = torch.ops.gs_ops.IndptrSplitByOffset(indptr, seeds_ptr)
            indt = torch.ops.gs_ops.SplitByOffset(indices, indices_ptr)
            # torch.cuda.nvtx.range_pop()

            for unique, indptr, indices in zip(unit, ptrt, indt):
                block = create_block_from_csc(indptr,
                                              indices,
                                              torch.tensor([]),
                                              num_src=unique.numel(),
                                              num_dst=indptr.numel() - 1)
                block.srcdata['_ID'] = unique
                pass
            seeds, seeds_ptr = unique_tensor, unique_tensor_ptr
            torch.cuda.synchronize()
            layer_end = time.time()
            if layer == 0:
                batch_layer_time_1 += layer_end - layer_start
            else:
                batch_layer_time_2 += layer_end - layer_start
        # torch.cuda.nvtx.range_pop()
    layer_time[0].append(batch_layer_time_1)
    layer_time[1].append(batch_layer_time_2)
    torch.cuda.synchronize()
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
    torch.cuda.synchronize()
    begin = time.time()
    batch_layer_time_1 = 0
    batch_layer_time_2 = 0
    for it, seeds in enumerate(tqdm(seedloader)):
        for layer, fanout in enumerate(fanouts):
            torch.cuda.synchronize()
            layer_start = time.time()
            subA = A._CAPI_fused_columnwise_slicing_sampling(
                seeds, fanout, False)
            unique_tensor, num_row, num_col, format_tensor1, format_tensor2, e_ids, format = subA._CAPI_relabel()
            block = create_block_from_csc(format_tensor1,
                                          format_tensor2,
                                          torch.tensor([]),
                                          num_src=num_row,
                                          num_dst=num_col)
            block.srcdata['_ID'] = unique_tensor
            seeds = unique_tensor
            torch.cuda.synchronize()
            layer_end = time.time()
            if layer == 0:
                batch_layer_time_1 += layer_end - layer_start
            else:
                batch_layer_time_2 += layer_end - layer_start
    layer_time[0].append(batch_layer_time_1)
    layer_time[1].append(batch_layer_time_2)
    torch.cuda.synchronize()
    end = time.time()
    time_list.append(end - begin)

print("w/o batching:", np.mean(time_list[2:]))
print("w/o batching layer1:", np.mean(layer_time[0][2:]))
print("w/o batching layer2:", np.mean(layer_time[1][2:]))
