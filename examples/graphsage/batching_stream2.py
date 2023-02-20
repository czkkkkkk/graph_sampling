import numpy as np
import gs
import torch
from gs.utils import create_block_from_csc
from gs.utils import SeedGenerator, load_reddit
import time
from tqdm import tqdm
from threading import Thread

torch.manual_seed(1)


def graphsage_sampler_thread(A, thread_seeds, fanouts, batch_size, num_streams):
    seedloader = SeedGenerator(thread_seeds, batch_size=batch_size,
                               shuffle=False, drop_last=False)
    ss = [torch.cuda.Stream() for i in range(num_streams)]
    for it, seeds in enumerate(tqdm(seedloader)):
        iter_sample = 0
        iter_relabel = 0
        s = ss[it % num_streams]
        with torch.cuda.stream(s):
            # torch.cuda.nvtx.range_push('sampler')
            for fanout in fanouts:
                s.synchronize()
                start = time.time()
                subA = A._CAPI_fused_columnwise_slicing_sampling(
                    seeds, fanout, False)
                s.synchronize()
                iter_sample += time.time() - start
                unique_tensor, num_row, num_col, format_tensor1, format_tensor2, e_ids, format = subA._CAPI_relabel()
                s.synchronize()
                iter_relabel += time.time() - start
                # block = create_block_from_csc(format_tensor1,
                #                               format_tensor2,
                #                               torch.tensor([]),
                #                               num_src=num_row,
                #                               num_dst=num_col)
                # block.srcdata['_ID'] = unique_tensor

                seeds = unique_tensor
            # torch.cuda.nvtx.range_pop()
        s.synchronize()
        sampling_time.append(iter_sample)
        relabel_time.append(iter_relabel)
    for s in ss:
        s.synchronize()


g, features, labels, n_classes, splitted_idx = load_reddit()
g = g.long().to('cuda')
train_nid = splitted_idx['train'].cuda()
val_nid = splitted_idx['valid'].cuda()
nid = torch.cat([train_nid, val_nid])
indptr, indices, _ = g.adj_sparse('csc')

n_epoch = 5
batch_size = 256
fanouts = [25, 15]
num_threads = 8
num_streams_per_thread = 16
thread_size = int(nid.numel() / num_threads)
print(batch_size, num_threads, num_streams_per_thread, thread_size)

A = gs.Graph(False)
A._CAPI_load_csc(indptr, indices)
torch.cuda.synchronize()

# graphsage (batch)
time_list = []
sampling_time = []
relabel_time = []
for epoch in range(n_epoch):
    # torch.cuda.nvtx.range_push('epoch')
    torch.cuda.synchronize()
    begin = time.time()
    threads = []
    for rank in range(num_threads):
        end = nid.numel() if rank == num_threads - 1 else (rank + 1) * thread_size
        t = Thread(target=graphsage_sampler_thread, args=(
            A, nid[rank * thread_size:end], fanouts, batch_size, num_streams_per_thread))
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
print('w/ batching _CAPI_fused_columnwise_slicing_sampling time per iteration:',
      np.mean(sampling_time))
print('w/ batching _CAPI_relabel time per iteration:', np.mean(relabel_time))

time_list = []
sampling_time = []
relabel_time = []
seedloader = SeedGenerator(nid, batch_size=batch_size,
                           shuffle=False, drop_last=False)
for epoch in range(n_epoch):
    torch.cuda.synchronize()
    begin = time.time()
    for it, seeds in enumerate(tqdm(seedloader)):
        iter_sample = 0
        iter_relabel = 0
        # torch.cuda.nvtx.range_push('sampler')
        for layer, fanout in enumerate(fanouts):
            torch.cuda.synchronize()
            start = time.time()
            subA = A._CAPI_fused_columnwise_slicing_sampling(
                seeds, fanout, False)
            torch.cuda.synchronize()
            iter_sample += time.time() - start
            unique_tensor, num_row, num_col, format_tensor1, format_tensor2, e_ids, format = subA._CAPI_relabel()
            torch.cuda.synchronize()
            iter_relabel += time.time() - start
            # block = create_block_from_csc(format_tensor1,
            #                               format_tensor2,
            #                               torch.tensor([]),
            #                               num_src=num_row,
            #                               num_dst=num_col)
            # block.srcdata['_ID'] = unique_tensor
            seeds = unique_tensor
        # torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        sampling_time.append(iter_sample)
        relabel_time.append(iter_relabel)
    torch.cuda.synchronize()
    end = time.time()
    print(end - begin)
    time_list.append(end - begin)

print("w/o batching:", np.mean(time_list[2:]))
print('w/o batching _CAPI_fused_columnwise_slicing_sampling time per iteration:',
      np.mean(sampling_time))
print('w/o batching _CAPI_relabel time per iteration:', np.mean(relabel_time))
