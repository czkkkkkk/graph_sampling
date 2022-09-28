import torch
import torch.nn as nn
import dgl.function as fn
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler
import tqdm
import torch.nn.functional as F


class GraphConv(nn.Module):
    def __init__(self, in_feats, n_classes):
        super().__init__()
        self.W = nn.Linear(in_feats, n_classes)

    def forward(self, g, x, w):
        with g.local_scope():
            g.srcdata['x'] = self.W(x)
            g.edata['w'] = w
            g.update_all(fn.u_mul_e('x', 'w', 'm'), fn.sum('m', 'y'))
            return g.dstdata['y']


class SAGEConv(nn.Module):
    def __init__(self, in_feats, n_classes):
        super().__init__()
        self.W = nn.Linear(in_feats * 2, n_classes)

    def forward(self, g, x, w):
        with g.local_scope():
            g.srcdata['x'] = x
            g.dstdata['x'] = x[:g.number_of_dst_nodes()]
            #g.edata['w'] = w
            #g.update_all(fn.u_mul_e('x', 'w', 'm'), fn.sum('m', 'y'))
            g.update_all(fn.copy_u('x', 'm'), fn.mean('m', 'y'))
            h = torch.cat([g.dstdata['x'], g.dstdata['y']], 1)
            return self.W(h)


class ConvModel(nn.Module):
    def __init__(self, in_size, hid_size, out_size, feat_device, conv=GraphConv):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(conv(in_size, hid_size))
        self.layers.append(conv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size
        self.feat_device = feat_device

    def forward(self, blocks, x):
        h = x.to('cuda') if self.feat_device == 'cpu' else x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size, feat):
        """Conduct layer-wise inference to get all the node embeddings."""
        sampler = MultiLayerFullNeighborSampler(1)
        dataloader = DataLoader(
            g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(), self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device, pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)
            feat = y
        return y
