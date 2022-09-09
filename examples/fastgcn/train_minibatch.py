import gs
from gs.jit.passes import dce
from gs import SeedGenerator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl.nn as dglnn
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler
import numpy as np
import time
import tqdm
import argparse
from load_graph import load_reddit
from dgl import DGLHeteroGraph, create_block


device = torch.device('cuda')
time_list = []


def fastgcn_sampler(A: gs.Matrix, seeds: torch.Tensor, probs: torch.Tensor, fanouts: list):
    output_nodes = seeds
    ret = []
    for fanout in fanouts:
        subA = A[:, seeds]
        row_indices = subA.row_indices()

        # selected, _ = torch.ops.gs_ops.list_sampling(row_indices,
        #                                              fanout, False)

        selected, _ = torch.ops.gs_ops.list_sampling_with_probs(
            row_indices, probs[row_indices], fanout, False)

        subA = subA[selected, :]
        seeds = subA.all_indices()
        ret.insert(0, subA.to_dgl_block())
    input_nodes = seeds
    return input_nodes, output_nodes, ret


def slicing_and_sampling_fuse(gm):
    """
    Fuses columnwise_slicing and columnwise_sampling
    """
    for node in gm.graph.nodes:
        if node.target == 'columnwise_sampling' and node.args[
                0].target == 'columnwise_slicing':
            if len(node.args[0].users) > 1:
                continue
            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_method(
                    'fused_columnwise_slicing_sampling',
                    args=(
                        *node.args[0].args,
                        *node.args[1:],
                    ))
                node.replace_all_uses_with(new_node)
    gm.graph.lint()
    gm.recompile()
    return gm


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size, feat_device):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))
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


def evaluate(model, matrix, compiled_func, seedloader, features, labels, probs):
    model.eval()
    ys = []
    y_hats = []
    for it, seeds in enumerate(seedloader):
        input_nodes, output_nodes, blocks = compiled_func(
            matrix, seeds, probs, [200, 200])
        with torch.no_grad():
            x = features[input_nodes]
            y = labels[output_nodes]
            ys.append(y)
            y_hats.append(model(blocks, x))
    return MF.accuracy(torch.cat(y_hats), torch.cat(ys))


def layerwise_infer(graph, nid, model, batch_size, feat, label):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size,
                               feat)  # pred in buffer_device
        pred = pred[nid]
        label = label[nid].to(pred.device)
        return MF.accuracy(pred, label)


def train(g, dataset, feat_device):
    features, labels, n_classes, train_idx, val_idx, test_idx = dataset
    model = SAGE(features.shape[1], 256, n_classes, feat_device).to(device)
    # create sampler & dataloader
    m = gs.Matrix(gs.Graph(False))
    m.load_dgl_graph(g)
    print("Check load successfully:", m._graph._CAPI_metadata(), '\n')
    fanouts = [200, 200]
    compiled_func = gs.jit.compile(
        func=fastgcn_sampler, args=(m, torch.Tensor(), torch.Tensor(), fanouts))
    compiled_func.gm = dce(slicing_and_sampling_fuse(compiled_func.gm))
    train_seedloader = SeedGenerator(
        train_idx, batch_size=1024, shuffle=True, drop_last=False)
    val_seedloader = SeedGenerator(
        val_idx, batch_size=1024, shuffle=True, drop_last=False)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    n_epoch = 10

    probs = g.out_degrees().float().cuda()

    for epoch in range(n_epoch):
        torch.cuda.synchronize()
        start = time.time()
        
        model.train()
        total_loss = 0
        for it, seeds in enumerate(train_seedloader):
            input_nodes, output_nodes, blocks = compiled_func(
                m, seeds, probs, fanouts)
            x = features[input_nodes]
            y = labels[output_nodes]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        
        torch.cuda.synchronize()
        time_list.append(time.time() - start)

        acc = evaluate(model, m, compiled_func,
                       val_seedloader, features, labels, probs)
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} "
              .format(epoch, total_loss / (it+1), acc.item()))
        print(torch.cuda.max_memory_reserved() / (1024 * 1024 * 1024), 'GB')

    print('Average epoch time:', np.mean(time_list[3:]))

    print('Testing...')
    acc = layerwise_infer(g, test_idx, model,
                          batch_size=4096, feat=features, label=labels)
    print("Test Accuracy {:.4f}".format(acc.item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmode", default='cuda', choices=['cpu', 'cuda'],
                        help="Feature reside device. To cpu or gpu")
    args = parser.parse_args()
    print(args)
    feat_device = args.fmode
    # load and preprocess dataset
    print('Loading data')
    g, features, labels, n_classes, splitted_idx = load_reddit()
    print('num of nodes:', g.num_nodes())
    print('num of edges:', g.num_edges())
    g = g.long().to('cuda')
    train_mask, val_mask, test_mask = splitted_idx['train'], splitted_idx['val'], splitted_idx['test']
    train_idx = train_mask.to(device)
    val_idx = val_mask.to(device)
    features = features.to(feat_device)
    labels = labels.to(device)

    train(g, (features, labels, n_classes, train_idx,
              val_idx, test_mask), feat_device)
