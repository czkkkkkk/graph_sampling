import torch
from torch.fx import Proxy

class Matrix(object):
    def __init__(self, graph: torch.classes.gs_classes.Graph):
        # Graph bind to a C++ object
        self._graph = graph

    def load_dgl_graph(self):
        pass

    def __getitem__(self, data):
        ret = self._graph
        r_slice = data[0]
        c_slice = data[1]
        if isinstance(r_slice, Proxy) or isinstance(r_slice, torch.Tensor):
            ret = ret.columnwise_slicing(r_slice)

        if isinstance(c_slice, Proxy) or isinstance(c_slice, torch.Tensor):
            ret = ret.columnwise_slicing(c_slice)

        return Matrix(ret)


