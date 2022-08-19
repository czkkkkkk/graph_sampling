from dataclasses import replace
import imp
import torch
import load_graph
import dgl
from dgl.transforms.functional import to_block
import time
import numpy as np

def bench(func, args):
    time_list = []
    for i in range(100):
        torch.cuda.synchronize()
        begin = time.time()
        
        func(*args)
        
        torch.cuda.synchronize()
        end = time.time()
        
        time_list.append(end - begin)
        
    print("AVG:", np.mean(time_list[10:]) * 1000)


device = torch.device('cuda:%d' % 0)

dataset = load_graph.load_reddit()
dgl_graph = dataset[0]
g = dgl_graph.long()
g = g.formats(['csc'])
g = g.to("cuda")
seeds = torch.arange(0, 5000).long().cuda()


def graphsage_baseline(g, seeds, fan_out):
    output_nodes = seeds
    seed_nodes = seeds
    blocks = []
    for num_pick in fan_out:
        frontier = g.sample_neighbors(seed_nodes, num_pick, replace=True)
        eid = frontier.edata['_ID']
        block = to_block(frontier, seed_nodes)
        block.edata['_ID'] = eid
        seed_nodes = block.srcdata['_ID']
        blocks.insert(0, block)

    return seed_nodes, output_nodes, blocks


print(graphsage_baseline(g, seeds, [25,15]))
bench(graphsage_baseline, args=(g, seeds, [25,15]))