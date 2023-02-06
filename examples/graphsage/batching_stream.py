import numpy as np
import gs
import torch
from gs.utils import create_block_from_csc
from gs.utils import SeedGenerator, load_reddit
import numpy as np
import time
from tqdm import tqdm
from threading import Thread

torch.manual_seed(1)


def graphsage_sampler(A, seeds, fanouts, rank):
    output_nodes = seeds
    ret = []
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for fanout in fanouts:
            subA = A._CAPI_fused_columnwise_slicing_sampling(
                seeds, fanout, False, rank)
            unique_tensor, num_row, num_col, format_tensor1, format_tensor2, e_ids, format = subA._CAPI_relabel(rank)
            block = create_block_from_csc(format_tensor1,
                                          format_tensor2,
                                          torch.tensor([]),
                                          num_src=num_row,
                                          num_dst=num_col)
            block.srcdata['_ID'] = unique_tensor
            seeds = unique_tensor
            ret.insert(0, block)
    input_nodes = seeds
    return input_nodes, output_nodes, ret


if __name__ == '__main__':
    g, features, labels, n_classes, splitted_idx = load_reddit()
    g = g.long().to('cuda')
    train_nid = splitted_idx['train'].cuda()
    val_nid = splitted_idx['valid'].cuda()
    nid = torch.cat([train_nid, val_nid])
    indptr, indices, _ = g.adj_sparse('csc')

    n_epoch = 5
    batch_size = 1024
    small_batch_size = 256
    fanouts = [25, 15]

    A = gs.Graph(False)
    A._CAPI_load_csc(indptr, indices)

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
            batch_seeds = torch.split(seeds, small_batch_size)
            num_batchs = len(batch_seeds)
            rets = []
            threads = []
            for rank in range(num_batchs):
                t = Thread(target=graphsage_sampler, args=(
                    A, batch_seeds[rank], fanouts, rank, ))
                t.start()
                threads.append(t)
            for t in threads:
                t.join()
            # exit()
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
