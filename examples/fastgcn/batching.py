from tqdm import tqdm
import time
from gs.utils import SeedGenerator, load_reddit, load_ogb
from gs.utils import create_block_from_coo
import torch
import gs
import numpy as np

torch.manual_seed(1)

g, features, labels, n_classes, splitted_idx = load_ogb(
    'ogbn-products', root='/home/ubuntu/gs-experiments/datasets')
g = g.long().to('cuda')
probs = g.out_degrees().float().cuda()
train_nid = splitted_idx['train'].cuda()
val_nid = splitted_idx['valid'].cuda()
nid = torch.cat([train_nid, val_nid])
indexes = torch.randperm(nid.shape[0], device=nid.device)
nid = nid[indexes].to('cuda')
indptr, indices, _ = g.adj_sparse('csc')

n_epoch = 5
batch_size = 51200
small_batch_size = 256
num_batchs = int((batch_size + small_batch_size - 1) / small_batch_size)
fanouts = [500, 500]
print(n_epoch, batch_size, small_batch_size, fanouts)

A = gs.Graph(False)
A._CAPI_load_csc(indptr, indices)
orig_seeds_ptr = torch.arange(num_batchs + 1, dtype=torch.int64,
                              device='cuda') * small_batch_size
orig_seeds_ptr[-1] = batch_size

# fastgcn (batch)
time_list = []
layer_time = [[], []]
seedloader = SeedGenerator(nid,
                           batch_size=batch_size,
                           shuffle=False,
                           drop_last=False)
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
            seeds_ptr = torch.arange(num_batchs + 1,
                                     dtype=torch.int64,
                                     device='cuda') * small_batch_size
            seeds_ptr[-1] = seeds.numel()

        for layer, fanout in enumerate(fanouts):
            torch.cuda.synchronize()
            layer_start = time.time()

            subA = A._CAPI_slicing(seeds, 0, gs._CSC, gs._CSC + gs._COO, False)

            indptr, indices, indices_ptr = subA.GetBatchCSC(seeds_ptr)

            neighbors, neighbors_ptr, neighbors_key = torch.ops.gs_ops.BatchUnique(
                indices, indices_ptr)

            node_probs = probs[neighbors]
            selected, _, selected_ptr = torch.ops.gs_ops.batch_list_sampling_with_probs(
                neighbors, node_probs, fanout, False, neighbors_ptr)

            # slicing

            coo_row, coo_col = subA.GetBatchCOO()
            sub_coo_row, sub_coo_col, sub_coo_ptr = torch.ops.gs_ops.BatchCOOSlicing(
                1, coo_row, coo_col, indices_ptr, selected, selected_ptr)

            # relabel
            mapping_data, mapping_data_key, mapping_data_ptr = torch.ops.gs_ops.BatchConcat(
                [seeds, sub_coo_row], [seeds_ptr, sub_coo_ptr])

            data, data_key, data_ptr = torch.ops.gs_ops.BatchConcat(
                [sub_coo_col, sub_coo_row], [sub_coo_ptr, sub_coo_ptr])

            unique_tensor, unique_tensor_ptr, relabel_data, relabel_data_ptr = torch.ops.gs_ops.BatchRelabelByKey(
                mapping_data, mapping_data_ptr, mapping_data_key, data,
                data_ptr, data_key)
            torch.ops.gs_ops.BatchSplit(relabel_data, relabel_data_ptr,
                                        data_key, [sub_coo_col, sub_coo_row],
                                        [sub_coo_ptr, sub_coo_ptr])

            seedst = torch.ops.gs_ops.SplitByOffset(seeds, seeds_ptr)
            unit = torch.ops.gs_ops.SplitByOffset(unique_tensor,
                                                  unique_tensor_ptr)
            colt = torch.ops.gs_ops.SplitByOffset(sub_coo_col, sub_coo_ptr)
            rowt = torch.ops.gs_ops.SplitByOffset(sub_coo_row, sub_coo_ptr)

            for s, unique, col, row in zip(seedst, unit, colt, rowt):
                block = create_block_from_coo(row,
                                              col,
                                              num_src=unique.numel(),
                                              num_dst=s.numel())
                block.srcdata['_ID'] = unique
            seeds, seeds_ptr = unique_tensor, unique_tensor_ptr
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

print("w/ batching:", np.mean(time_list[2:]))
print("w/ batching layer1:", np.mean(layer_time[0][2:]))
print("w/ batching layer2:", np.mean(layer_time[1][2:]))

time_list = []
layer_time = [[], []]
seedloader = SeedGenerator(nid,
                           batch_size=small_batch_size,
                           shuffle=False,
                           drop_last=False)
for epoch in range(n_epoch):
    torch.cuda.synchronize()
    begin = time.time()
    batch_layer_time_1 = 0
    batch_layer_time_2 = 0
    for it, seeds in enumerate(tqdm(seedloader)):
        for layer, fanout in enumerate(fanouts):
            torch.cuda.synchronize()
            layer_start = time.time()
            subA = A._CAPI_slicing(seeds, 0, gs._CSC, gs._CSC + gs._COO, False)
            neighbors = subA._CAPI_get_valid_rows()
            node_probs = probs[neighbors]
            selected, _ = torch.ops.gs_ops.list_sampling_with_probs(
                neighbors, node_probs, fanout, False)
            subA = subA._CAPI_slicing(selected, 1, gs._COO, gs._COO, False)
            unique_tensor, num_row, num_col, format_tensor1, format_tensor2, e_ids, format = subA._CAPI_relabel(
            )
            block = create_block_from_coo(format_tensor1,
                                          format_tensor2,
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
