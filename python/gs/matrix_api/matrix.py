import torch

class Matrix(object):
    def __init__(self, graph: torch.classes.gs_classes.Graph):
        # Graph bind to a C++ object
        self._graph = graph

    def load_dgl_graph(self):
        pass

    def __getitem__(self, data):
        # assert isinstance(data, tuple), 'data type is {}'.format(type(data))
        # r_slice = data[0]
        c_slice = data[1]
        # Columnwise Slicing
        # if r_slice == slice(None, None, None) and isinstance(c_slice, torch.Tensor):
        return Matrix(self._graph.columnwise_slicing(c_slice))
        # else:
        #     pass


