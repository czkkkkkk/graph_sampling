#include <torch/custom_class.h>
#include <torch/script.h>

#include "./graph.h"
#include "./graph_ops.h"
#include "./hetero_graph.h"
#include "./tensor_ops.h"
using namespace gs;

TORCH_LIBRARY(gs_classes, m) {
  m.class_<Graph>("Graph")
      .def(torch::init<bool>())
      .def("columnwise_slicing", &Graph::ColumnwiseSlicing)
      .def("rowwise_slicing", &Graph::RowwiseSlicing)
      .def("columnwise_sampling", &Graph::ColumnwiseSampling)
      .def("fused_columnwise_slicing_sampling",
           &Graph::ColumnwiseFusedSlicingAndSampling)
      .def("load_csc", &Graph::LoadCSC)
      .def("load_csc_with_col_ids", &Graph::LoadCSCWithColIds)
      .def("row_indices", &Graph::RowIndices)
      .def("print", &Graph::Print)
      .def("all_indices", &Graph::AllIndices)
      .def("relabel", &Graph::Relabel)
      .def("_CAPI_metadata", &Graph::MetaData)
      .def("random_walk",&Graph::RandomWalk);
  m.class_<HeteroGraph>("HeteroGraph")
      .def(torch::init<>())
      .def("load_from_homo", &HeteroGraph::LoadFromHomo)
      .def("get_homo_graph", &HeteroGraph::GetHomoGraph)
      .def("metapath_random_walk_fused", &HeteroGraph::MetapathRandomWalkFused);
}

TORCH_LIBRARY(gs_ops, m) {
  m.def("list_sampling", &ListSampling);
  m.def("list_sampling_with_probs", &ListSamplingProbs);
}

namespace gs {}