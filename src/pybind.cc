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
      .def("_CAPI_set_data", &Graph::SetData)
      .def("_CAPI_get_data", &Graph::GetData)
      .def("_CAPI_get_num_rows", &Graph::GetNumRows)
      .def("_CAPI_get_num_cols", &Graph::GetNumCols)
      .def("_CAPI_columnwise_slicing", &Graph::ColumnwiseSlicing)
      .def("_CAPI_rowwise_slicing", &Graph::RowwiseSlicing)
      .def("_CAPI_columnwise_sampling", &Graph::ColumnwiseSampling)
      .def("_CAPI_fused_columnwise_slicing_sampling",
           &Graph::ColumnwiseFusedSlicingAndSampling)
      .def("_CAPI_load_csc", &Graph::LoadCSC)
      .def("_CAPI_load_csc_with_col_ids", &Graph::LoadCSCWithColIds)
      .def("_CAPI_all_valid_node", &Graph::AllValidNode)
      .def("_CAPI_get_rows", &Graph::GetRows)
      .def("_CAPI_get_cols", &Graph::GetCols)
      .def("_CAPI_get_valid_rows", &Graph::GetValidRows)
      .def("_CAPI_get_valid_cols", &Graph::GetValidCols)
      .def("_CAPI_get_coo_rows", &Graph::GetCOORows)
      .def("_CAPI_get_coo_cols", &Graph::GetCOOCols)
      .def("_CAPI_relabel", &Graph::Relabel)
      .def("_CAPI_sum", &Graph::Sum)
      .def("_CAPI_divide", &Graph::Divide)
      .def("_CAPI_normalize", &Graph::Normalize)
      .def("_CAPI_metadata", &Graph::MetaData)
      .def("_CAPI_random_walk", &Graph::RandomWalk);
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