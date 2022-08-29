import dgl
import torch
from dgl.data import RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset
import scipy.sparse as sp


def load_reddit():
    data = RedditDataset(self_loop=True)
    g = data[0]
    n_classes = data.num_classes
    train_nid = torch.nonzero(g.ndata["train_mask"], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata["test_mask"], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata["val_mask"], as_tuple=True)[0]
    splitted_idx = {"train": train_nid, "test": test_nid, "val": val_nid}
    feat = g.ndata['feat']
    labels = g.ndata['label']
    g.ndata.clear()
    return g, feat, labels, n_classes, splitted_idx


def load_ogbn_products():
    data = DglNodePropPredDataset(name="ogbn-products", root="../datasets")
    splitted_idx = data.get_idx_split()
    g, labels = data[0]
    feat = g.ndata['feat']
    labels = labels[:, 0]
    n_classes = len(
        torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
    g.ndata.clear()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g, feat, labels, n_classes, splitted_idx


import os
from ctypes import *
from ctypes.util import *
import numpy as np
libgraphPath = '/home/ubuntu/gs-experiments/nextdoor-dgl/libgraph.so'
libgraph = CDLL(libgraphPath)
libgraph.loadgraph.argtypes = [c_char_p]
def load_custom_reddit(filename):
    if not os.path.exists(filename):
        raise Exception("'%s' do not exist" % (filename))

    graphPath = bytes(filename, encoding='utf8')
    libgraph.loadgraph(graphPath)
    libgraph.getEdgePairList.restype = np.ctypeslib.ndpointer(
        dtype=c_int, shape=(libgraph.numberOfEdges(), 2))

    print("Graph Loaded in C++")

    edges = libgraph.getEdgePairList()
    print("Number of Edges", libgraph.numberOfEdges())
    print("Number of Vertices", libgraph.numberOfVertices())
    src_ids = torch.tensor(edges[:, 0])
    dst_ids = torch.tensor(edges[:, 1])
    dgl_graph = dgl.graph((src_ids, dst_ids), idtype=torch.int64)
    num_nodes = dgl_graph.num_nodes()
    
    data = RedditDataset(self_loop=True)
    g = data[0]
    n_classes = data.num_classes
    train_nid = torch.nonzero(g.ndata["train_mask"][:num_nodes], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata["test_mask"][:num_nodes], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata["val_mask"][:num_nodes], as_tuple=True)[0]
    splitted_idx = {"train": train_nid, "test": test_nid, "val": val_nid}
    feat = g.ndata['feat'][:num_nodes].clone()
    labels = g.ndata['label'][:num_nodes]
    g.ndata.clear()
    return dgl_graph, feat, labels, n_classes, splitted_idx