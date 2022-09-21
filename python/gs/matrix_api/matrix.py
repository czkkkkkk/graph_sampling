import torch
from torch.fx import Proxy
import dgl
from dgl import DGLHeteroGraph, create_block

torch.fx.wrap('create_block')


class Matrix(object):

    def __init__(self, graph: torch.classes.gs_classes.Graph):
        # Graph bind to a C++ object
        self._graph = graph

    def load_dgl_graph(self, g):
        # import csc
        if not isinstance(g, DGLHeteroGraph):
            raise ValueError
        reverse_g = dgl.reverse(g)
        reverse_g = reverse_g.formats(['csr'])
        csc = reverse_g.adj(scipy_fmt='csr')
        csc_indptr = torch.tensor(csc.indptr).long().cuda()
        csc_indices = torch.tensor(csc.indices).long().cuda()
        self._graph.load_csc(csc_indptr, csc_indices)

    def to_dgl_block(self):
        unique_tensor, csc_indptr, csc_indices = self._graph.relabel()
        return create_block(('csc', (csc_indptr, csc_indices, [])),
                            num_src_nodes=unique_tensor.numel(),
                            num_dst_nodes=csc_indptr.numel() - 1)

    def columnwise_slicing(self, t):
        return Matrix(self._graph.columnwise_slicing(t))

    def columnwise_sampling(self, fanout, replace=True):
        return Matrix(self._graph.columnwise_sampling(fanout, replace))

    def row_indices(self) -> torch.Tensor:
        return self._graph.row_indices()

    def all_indices(self) -> torch.Tensor:
        return self._graph.all_indices()

    def row_indices(self, unique: bool = True) -> torch.Tensor:
        return self._graph.row_indices(unique)

    def __getitem__(self, data):
        ret = self._graph
        r_slice = data[0]
        c_slice = data[1]

        if isinstance(c_slice, Proxy) or isinstance(c_slice, torch.Tensor):
            ret = ret.columnwise_slicing(c_slice)

        if isinstance(r_slice, Proxy) or isinstance(r_slice, torch.Tensor):
            ret = ret.rowwise_slicing(r_slice)

        return Matrix(ret)

    def fused_columnwise_slicing_sampling(self, seeds, fanouts, raplace):
        return Matrix(self._graph.fused_columnwise_slicing_sampling(seeds, fanouts, raplace))
