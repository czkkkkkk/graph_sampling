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
    n_classes = len(torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
    g.ndata.clear()
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    return g, feat, labels, n_classes, splitted_idx