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
      .def("_CAPI_fusion_slicing", &Graph::FusedBidirSlicing)
      .def("_CAPI_get_num_rows", &Graph::GetNumRows)
      .def("_CAPI_get_num_cols", &Graph::GetNumCols)
      .def("_CAPI_get_num_edges", &Graph::GetNumEdges)
      .def("_CAPI_slicing", &Graph::Slicing)
      .def("_CAPI_sampling", &Graph::Sampling)
      .def("_CAPI_sampling_with_probs", &Graph::SamplingProbs)
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
      .def("_CAPI_random_walk", &Graph::RandomWalk)
      .def("_CAPI_sddmm", &Graph::SDDMM)
      .def("_CAPI_full_get_num_nodes", &Graph::GetNumNodes)
      .def("_CAPI_full_load_csc", &Graph::FullLoadCSC)
      .def("_CAPI_full_slicing", &Graph::FullSlicing)
      .def("_CAPI_full_sampling", &Graph::FullSampling)
      .def("_CAPI_full_sampling_with_probs", &Graph::FullSamplingProbs)
      .def("_CAPI_full_sum", &Graph::FullSum)
      .def("_CAPI_full_divide", &Graph::FullDivide)
      .def("_CAPI_full_normalize", &Graph::FullNormalize)
      .def("_CAPI_full_sddmm", &Graph::FullSDDMM)
      .def("_CAPI_get_coo", &Graph::FullGetCOO)
      .def("_CAPI_full_relabel", &Graph::FullRelabel);
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