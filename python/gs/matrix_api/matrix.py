import torch
from torch.fx import Proxy
import dgl
from dgl import DGLHeteroGraph, create_block


torch.fx.wrap('create_block')


class Matrix(object):

    def __init__(self, graph: torch.classes.gs_classes.Graph):
        # Graph bind to a C++ object
        self._graph = graph

    def set_data(self, data):
        self._graph._CAPI_set_data(data)

    def load_dgl_graph(self, g, weight=None):
        # import csc
        if not isinstance(g, DGLHeteroGraph):
            raise ValueError
        reverse_g = dgl.reverse(g)
        reverse_g = reverse_g.formats(['csr'])
        csc = reverse_g.adj(scipy_fmt='csr')
        csc_indptr = torch.tensor(csc.indptr).long().cuda()
        csc_indices = torch.tensor(csc.indices).long().cuda()
        self._graph._CAPI_load_csc(csc_indptr, csc_indices)
        if weight is not None:
            self._graph._CAPI_set_data(g.edata[weight])

    def to_dgl_block(self):
        unique_tensor, num_row, num_col, format_tensor1, format_tensor2, e_ids, format = self._graph._CAPI_relabel(
        )
        block = None
        if format == 'coo':
            block = create_block((format, (format_tensor1, format_tensor2)),
                                 num_src_nodes=num_row,
                                 num_dst_nodes=num_col)
        else:
            block = create_block(
                (format, (format_tensor1, format_tensor2, [])),
                num_src_nodes=num_row,
                num_dst_nodes=num_col)

        data = self._graph._CAPI_get_data()
        if data is not None:
            if e_ids is not None:
                data = data[e_ids]
            block.edata['w'] = data
        block.srcdata['_ID'] = unique_tensor
        return block

    def columnwise_slicing(self, t):
        return Matrix(self._graph._CAPI_columnwise_slicing(t))

    def columnwise_sampling(self, fanout, replace=True):
        return Matrix(self._graph._CAPI_columnwise_sampling(fanout, replace))

    def sum(self, axis, powk=1) -> torch.Tensor:
        return self._graph._CAPI_sum(axis, powk)

    def divide(self, divisor, axis):
        return Matrix(self._graph._CAPI_divide(divisor, axis))

    def normalize(self, axis):
        return Matrix(self._graph._CAPI_normalize(axis))

    def row_ids(self, unique=True) -> torch.Tensor:
        if unique:
            return self._graph._CAPI_get_valid_rows()
        else:
            return self._graph._CAPI_get_rows()

    def all_indices(self) -> torch.Tensor:
        return self._graph._CAPI_all_valid_node()

    def row_indices(self) -> torch.Tensor:
        return self._graph._CAPI_get_coo_rows(False)

    def __getitem__(self, data):
        ret = self._graph
        r_slice = data[0]
        c_slice = data[1]

        if isinstance(c_slice, Proxy) or isinstance(c_slice, torch.Tensor):
            ret = ret._CAPI_columnwise_slicing(c_slice)

        if isinstance(r_slice, Proxy) or isinstance(r_slice, torch.Tensor):
            ret = ret._CAPI_rowwise_slicing(r_slice)

        return Matrix(ret)

    def fused_columnwise_slicing_sampling(self, seeds, fanouts, raplace):
        return Matrix(
            self._graph._CAPI_fused_columnwise_slicing_sampling(
                seeds, fanouts, raplace))

    def random_walk(self, seeds, walk_length):
        return self._graph._CAPI_random_walk(seeds, walk_length)

    def relabel(self):
        self._graph._CAPI_relabel()
        return self
