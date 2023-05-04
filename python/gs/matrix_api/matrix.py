import torch
from torch.fx import Proxy
from dgl import DGLHeteroGraph
from typing import Optional, List
from gs.utils import create_block_from_coo, create_block_from_csc
from gs.format import _COO, _CSC, _CSR

torch.fx.wrap('create_block_from_coo')
torch.fx.wrap('create_block_from_csc')


class Matrix(object):

    def __init__(self):
        self._graph = None
        self.null_tensor = torch.Tensor().cuda().long()
        self.rows = self.null_tensor
        self.cols = self.null_tensor
        self.row_ndata = {}
        self.col_ndata = {}
        self.edata = {}

    def load_graph(self, format: str, format_tensors: List[torch.Tensor]):
        assert format in ['CSC', 'COO', 'CSR']
        assert len(format_tensors) == 2

        if format == 'COO':
            coo_row, coo_col = format_tensors
            self.num_rows = coo_row.max() + 1
            self.num_cols = coo_col.max() + 1
            self._graph = torch.classes.gs_classes.Graph(
                self.num_rows, self.num_cols)
            self._graph._CAPI_LoadCOO(coo_row, coo_col, False, False)

        elif format == 'CSC':
            indptr, indices = format_tensors
            self.num_rows = indices.max() + 1
            self.num_cols = indptr.numel()
            self._graph = torch.classes.gs_classes.Graph(
                self.num_rows, self.num_cols)
            self._graph._CAPI_LoadCSC(indptr, indices)

        elif format == 'CSR':
            indptr, indices = format_tensors
            self.num_rows = indptr.numel()
            self.num_cols = indices.max() + 1
            self._graph = torch.classes.gs_classes.Graph(
                self.num_rows, self.num_cols)
            self._graph._CAPI_LoadCSR(indptr, indices)
   

    def set_row_data(self, key, value):
        self.row_ndata[key] = value

    def set_col_data(self, key, value):
        self.col_ndata[key] = value

    def set_edge_data(self, key, value):
        self.edata[key] = value

    def _set_graph(self, graph: torch.classes.gs_classes.Graph):
        self._graph = graph

    def __getitem__(self, data):
        assert len(data) == 2
        ret = self._graph
        r_slice = data[0]
        c_slice = data[1]

        ret_matrix = Matrix()

        edge_index = None
        graph = self._graph

        if isinstance(c_slice, Proxy) or isinstance(c_slice, torch.Tensor):
            if self.cols.shape != torch.Size([0]):
                ret_matrix.cols = self.cols[c_slice]
            else:
                ret_matrix.cols = c_slice

            graph, _edge_index = graph._CAPI_Slicing(c_slice, 1, _CSC, _COO)
            edge_index = _edge_index

            for key, value in self.col_ndata.items():
                ret_matrix.set_col_data(key, value[c_slice])

        if isinstance(r_slice, Proxy) or isinstance(r_slice, torch.Tensor):
            if self.rows.shape != torch.Size([0]):
                ret_matrix.rows = self.rows[r_slice]
            else:
                ret_matrix.rows = r_slice

            graph, _edge_index = graph._CAPI_Slicing(r_slice, 0, _CSR, _COO)
            if edge_index is not None:
                edge_index = edge_index[_edge_index]

            for key, value in self.col_ndata.items():
                ret_matrix.set_col_data(key, value[r_slice])

        ret_matrix._set_graph(graph)
        for key, value in self.edata.items():
            ret_matrix.set_edge_data(key, value[_edge_index])

        return ret_matrix

    def individual_sampling(self, K: int, probs: torch.Tensor, replace: bool):
        ret_matrix = Matrix()

        if probs is None:
            subgraph, edge_index = self._graph._CAPI_Sampling(
                0, K, replace, _CSC, _COO)
        else:
            subgraph, edge_index = self._graph._CAPI_SamplingProbs(
                0, probs, K, replace, _CSC, _COO)

        ret_matrix._set_graph(subgraph)
        ret_matrix.cols = self.cols
        ret_matrix.rows = self.rows
        ret_matrix.row_ndata = self.row_ndata
        ret_matrix.col_ndata = self.col_ndata
        for key, value in self.edata.items():
            ret_matrix.set_edge_data(key, value[edge_index])

        return ret_matrix

    def to_dgl_block(self):
        unique_tensor, num_row, num_col, format_tensor1, format_tensor2, e_ids, format = self._graph._CAPI_GraphRelabel(
            self.cols, self.rows)

        if format == 'coo':
            block = create_block_from_coo(format_tensor1,
                                          format_tensor2,
                                          num_src=num_row,
                                          num_dst=num_col)
        else:
            block = create_block_from_csc(format_tensor1,
                                          format_tensor2,
                                          torch.tensor([]),
                                          num_src=num_row,
                                          num_dst=num_col)

        if e_ids is not None:
            block.edata['_ID'] = e_ids

        for key, value in self.edata.items():
            block.edata[key] = value
        block.srcdata['_ID'] = unique_tensor
        return block

    def all_nodes(self):
        return self._graph._CAPI_GetValidNodes(self.cols, self.rows)
