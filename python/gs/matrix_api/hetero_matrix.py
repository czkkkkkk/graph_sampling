
import torch
from .matrix import Matrix


class HeteroMatrix(object):
    def __init__(self, hetero_graph: torch.classes.gs_classes.HeteroGraph):
        self._hetero_graph = hetero_graph

    def load_from_homo(self, node_types, edge_types, homo_matrices):
        homo_graphs = [m._graph for m in homo_matrices]
        self._hetero_graph.load_from_homo(node_types, edge_types, homo_graphs)

    def get_homo_matrix(self, etype: str):
        return Matrix(self._hetero_graph.get_homo_graph(etype))

    def metapath_random_walk(self, seeds, metapath):
        return self._hetero_graph.metapath_random_walk(seeds, metapath)

    def metapath_random_walk_fused(self, seeds, metapath):
        return self._hetero_graph.metapath_random_walk_fused(seeds, metapath)
