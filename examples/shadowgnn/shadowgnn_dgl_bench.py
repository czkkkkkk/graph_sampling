from dataclasses import replace
import imp
from sys import meta_path
import torch
import load_graph
import dgl
from dgl.transforms.functional import to_block
from dgl.sampling import random_walk, pack_traces
import time
import numpy as np

device = torch.device('cuda:%d' % 0)

dataset = load_graph.load_reddit()
dgl_graph = dataset[0]
g = dgl_graph.long()
g = g.to("cuda")

#


def shadowgnn_baseline(graph: dgl.DGLGraph, seed_nodes, fanouts):
    torch.cuda.nvtx.range_push("shadowgnn baseline")
    output_nodes = seed_nodes
    for fanout in reversed(fanouts):
        frontier = graph.sample_neighbors(seed_nodes, fanout)
        block = dgl.transforms.to_block(frontier, seed_nodes)
        seed_nodes = block.srcdata[dgl.NID]
    subg = graph.subgraph(seed_nodes, relabel_nodes=True)
    torch.cuda.nvtx.range_pop()
    return seed_nodes, output_nodes, subg


str_list = []


def bench(loop_num, func, args):
    time_list = []
    for i in range(loop_num):
        torch.cuda.synchronize()
        begin = time.time()
        seed_nodes, output_nodes, subg = func(*args)
        #print("ret nodes:", seed_nodes.numel())
        torch.cuda.synchronize()
        end = time.time()

        time_list.append(end - begin)
    # str_list.append("%d,%d,%.3f" %
    #                 (seed_num, metalength, np.mean(time_list[10:]) * 1000))
    print("dgl shadowgnn AVG:", np.mean(time_list[10:]) * 1000, " ms.")


# seeds_set = [1000, 10000, 50000, 100000, 200000, 2000000, 10000000]
# metapath_len = [5, 10, 15, 20, 25, 30]


fanouts = [5, 15, 25]
#seeds = torch.randint(0, 232965, (5,), device='cuda')
seeds = torch.arange(0, 5).long().cuda()
bench(
    100,
    shadowgnn_baseline, args=(
        g,
        seeds,
        fanouts))
