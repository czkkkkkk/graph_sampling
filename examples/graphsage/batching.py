import gs
from gs.utils import SeedGenerator, load_reddit, ConvModel
import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
import numpy as np
import time
import argparse
from tqdm import tqdm


device = torch.device('cuda')
time_list = []
sampling_list = []
split_list = []
relabel_list = []


def graphsage_sampler(A: gs.Matrix, seeds, fanouts):
    output_nodes = seeds
    ret = []
    for fanout in fanouts:
        torch.cuda.synchronize()
        start = time.time()
        subA = A.fused_columnwise_slicing_sampling(seeds, fanout, True)
        torch.cuda.synchronize()
        sampling_list[-1] += time.time() - start
        start = time.time()
        block = subA.to_dgl_block()
        torch.cuda.synchronize()
        relabel_list[-1] += time.time() - start
        seeds = block.srcdata['_ID']
        ret.insert(0, block)
    input_nodes = seeds
    return input_nodes, output_nodes, ret


def graphsage_sampler_batching(A: gs.Matrix, seeds, fanouts):
    # torch.cuda.nvtx.range_push('sampler')
    output_nodes = seeds
    ret = []
    for fanout in fanouts:
        torch.cuda.synchronize()
        start = time.time()
        subA = A.fused_columnwise_slicing_sampling(seeds, fanout, True)
        torch.cuda.synchronize()
        sampling_list[-1] += time.time() - start
        start = time.time()
        # torch.cuda.nvtx.range_push('split')
        batchAs = subA._graph._CAPI_split(256)
        torch.cuda.synchronize()
        split_list[-1] += time.time() - start
        # torch.cuda.nvtx.range_pop()
        # torch.cuda.nvtx.range_push('toblock')
        for bachA in batchAs:
            # torch.cuda.nvtx.range_push('single')
            torch.cuda.synchronize()
            start = time.time()
            block = gs.Matrix(bachA).to_dgl_block()
            torch.cuda.synchronize()
            relabel_list[-1] += time.time() - start
            # torch.cuda.nvtx.range_pop()
            seeds = block.srcdata['_ID']
            ret.insert(0, block)
        # torch.cuda.nvtx.range_pop()
    input_nodes = seeds
    # torch.cuda.nvtx.range_pop()
    return input_nodes, output_nodes, ret


def train(g, dataset, feat_device):
    features, labels, n_classes, train_idx, val_idx, test_idx = dataset
    model = ConvModel(features.shape[1], 256,
                      n_classes, feat_device).to(device)
    # create sampler & dataloader
    m = gs.Matrix(gs.Graph(False))
    m.load_dgl_graph(g)
    print("Check load successfully:", m._graph._CAPI_metadata(), '\n')
    compiled_func = graphsage_sampler_batching if args.batching else graphsage_sampler
    train_seedloader = SeedGenerator(
        train_idx, batch_size=args.batchsize, shuffle=True, drop_last=False)
    val_seedloader = SeedGenerator(
        val_idx, batch_size=args.batchsize, shuffle=True, drop_last=False)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    n_epoch = 5

    for epoch in range(n_epoch):
        sampling_list.append(0)
        split_list.append(0)
        relabel_list.append(0)
        torch.cuda.synchronize()
        start = time.time()
        model.train()
        total_loss = 0
        for it, seeds in enumerate(tqdm(train_seedloader)):
            input_nodes, output_nodes, blocks = compiled_func(
                m, seeds, args.fanout)

        model.eval()
        ys = []
        y_hats = []
        for it, seeds in enumerate(tqdm(val_seedloader)):
            input_nodes, output_nodes, blocks = compiled_func(
                m, seeds, args.fanout)
        torch.cuda.synchronize()
        time_list.append(time.time() - start)

    print('Average epoch time:', np.mean(time_list[3:]))
    print('Average epoch sample time:', np.mean(sampling_list[3:]))
    print('Average epoch split time:', np.mean(split_list[3:]))
    print('Average epoch relabel time:', np.mean(relabel_list[3:]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmode", default='cuda', choices=['cpu', 'cuda'],
                        help="Feature reside device. To cpu or gpu")
    parser.add_argument("--batchsize", type=int, default=256,
                        help="batch size for training")
    parser.add_argument("--fanout", default='10',
                        help="sample size for each layer")
    args = parser.parse_args()
    args.batching = False
    print(args)
    args.fanout = [int(x.strip()) for x in args.fanout.split(',')]
    feat_device = args.fmode
    # load and preprocess dataset
    print('Loading data')
    g, features, labels, n_classes, splitted_idx = load_reddit()
    g = g.to('cuda')
    train_mask, val_mask, test_mask = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    train_idx = train_mask.to(device)
    val_idx = val_mask.to(device)
    features = features.to(feat_device)
    labels = labels.to(device)

    train(g, (features, labels, n_classes, train_idx,
              val_idx, test_mask), feat_device)
