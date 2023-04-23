#include <torch/custom_class.h>
#include <torch/script.h>

#include "./graph.h"
#include "./graph_ops.h"
#include "./hetero_graph.h"
#include "./tensor_ops.h"
#include "cuda/tensor_ops.h"
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
      .def("_CAPI_load_coo", &Graph::LoadCOO)
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
      .def("_CAPI_normalize", &Graph::Normalize)
      .def("_CAPI_divide", &Graph::Divide)
      .def("_CAPI_metadata", &Graph::MetaData)
      .def("_CAPI_coo_metadata", &Graph::COOMetaData)
      .def("_CAPI_csc_metadata", &Graph::CSCMetaData)
      .def("_CAPI_random_walk", &Graph::RandomWalk)
      .def("_CAPI_sddmm", &Graph::SDDMM)
      .def("_CAPI_split", &Graph::Split)
      .def("_CAPI_get_csc", &Graph::GetCSCTensor)
      .def("_CAPI_get_coo", &Graph::GetCOOTensor)
      // .def("_CAPI_batch_slicing", &Graph::BatchSlicing)
      .def("_CAPI_set_coo", &Graph::SetCOOByTensor)
      .def("_CAPI_batch_slicing", &Graph::BatchColSlicing)
      .def("_CAPI_decode", &Graph::Decode)
      .def("_CAPI_set_metadata", &Graph::SetMetaData)
      .def("_CAPI_batch_fusion_slicing", &Graph::BatchFusedBidirSlicing)
      .def("_CAPI_e_div_u_sum", &Graph::EDivUSum);

  m.class_<HeteroGraph>("HeteroGraph")
      .def(torch::init<>())
      .def("load_from_homo", &HeteroGraph::LoadFromHomo)
      .def("get_homo_graph", &HeteroGraph::GetHomoGraph)
      .def("metapath_random_walk_fused", &HeteroGraph::MetapathRandomWalkFused);
}

TORCH_LIBRARY(gs_ops, m) {
  m.def("list_sampling", &ListSampling);
  m.def("list_sampling_with_probs", &ListSamplingProbs);
  m.def("batch_list_sampling_with_probs", &BatchListSamplingProbs);
  m.def("index_search", &IndexSearch);
  m.def("SplitByOffset", &SplitByOffset);
  m.def("IndptrSplitBySize", &gs::impl::SplitIndptrBySizeCUDA);
  m.def("IndptrSplitByOffset", &gs::impl::SplitIndptrByOffsetCUDA);
  m.def("BatchConcat", &gs::impl::BatchConcatCUDA);
  m.def("BatchUnique", &gs::impl::BatchUniqueCUDA);
  m.def("BatchUniqueByKey", &gs::impl::BatchUniqueByKeyCUDA);
  m.def("BatchUnique2", &gs::impl::BatchUnique2CUDA);
  m.def("BatchUniqueByKey2", &gs::impl::BatchUniqueByKey2CUDA);
  m.def("BatchCSRRelabelByKey", &gs::impl::BatchCSRRelabelByKeyCUDA);
  m.def("BatchCSRRelabel", &gs::impl::BatchCSRRelabelCUDA);
  m.def("BatchCOORelabelByKey", &gs::impl::BatchCOORelabelByKeyCUDA);
  m.def("BatchCOORelabel", &gs::impl::BatchCOORelabelCUDA);
  m.def("BatchSplit", &gs::impl::BatchSplit2CUDA);
  m.def("BatchCOOSlicing", &gs::impl::BatchCOOSlicingCUDA);
  m.def("BatchEncode", &gs::impl::BatchEncodeCUDA);
  m.def("BatchDecode", &gs::impl::BatchDecodeCUDA);
  m.def("GetBatchOffsets", &gs::impl::GetBatchOffsets);
  m.def("COORowSlicingGlobalId", &gs::impl::COORowSlicingGlobalIdCUDA);
  m.def("_CAPI_unique", &gs::impl::TensorUniqueCUDA);
}

namespace gs {}