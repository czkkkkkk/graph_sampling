import numpy as np
import gs
import torch
from gs.utils import create_block_from_csc
from gs.utils import SeedGenerator, load_reddit
import time
from tqdm import tqdm
from threading import Thread

torch.manual_seed(1)


def graphsage_sampler(A, seeds, fanouts):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        # torch.cuda.nvtx.range_push('sampler')
        for fanout in fanouts:
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
        # torch.cuda.nvtx.range_pop()
    s.synchronize()


g, features, labels, n_classes, splitted_idx = load_reddit()
g = g.long().to('cuda')
train_nid = splitted_idx['train'].cuda()
val_nid = splitted_idx['valid'].cuda()
nid = torch.cat([train_nid, val_nid])
indptr, indices, _ = g.adj_sparse('csc')

n_epoch = 5
batch_size = 65536
small_batch_size = 256
fanouts = [25, 15]

A = gs.Graph(False)
A._CAPI_load_csc(indptr, indices)
torch.cuda.synchronize()

# graphsage (batch)
time_list = []
seedloader = SeedGenerator(nid, batch_size=batch_size,
                           shuffle=False, drop_last=False)
for epoch in range(n_epoch):
    # torch.cuda.nvtx.range_push('epoch')
    torch.cuda.synchronize()
    begin = time.time()
    threads = []
    for it, seeds in enumerate(tqdm(seedloader)):
        batch_seeds = torch.split(seeds, small_batch_size)
        num_batchs = len(batch_seeds)
        for rank in range(num_batchs):
            t = Thread(target=graphsage_sampler, args=(
                A, batch_seeds[rank], fanouts,))
            t.start()
            threads.append(t)
    for t in threads:
        t.join()
    torch.cuda.synchronize()
    end = time.time()
    print(end - begin)
    # torch.cuda.nvtx.range_pop()
    time_list.append(end - begin)

print("w/ batching:", np.mean(time_list[2:]))

time_list = []
seedloader = SeedGenerator(nid, batch_size=small_batch_size,
                           shuffle=False, drop_last=False)
for epoch in range(n_epoch):
    torch.cuda.synchronize()
    begin = time.time()
    for it, seeds in enumerate(tqdm(seedloader)):
        # torch.cuda.nvtx.range_push('sampler')
        for layer, fanout in enumerate(fanouts):
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
        # torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    end = time.time()
    print(end - begin)
    time_list.append(end - begin)

print("w/o batching:", np.mean(time_list[2:]))
