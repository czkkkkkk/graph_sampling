#include <torch/custom_class.h>
#include <torch/script.h>

#include "./graph.h"
#include "./tensor_ops.h"

using namespace gs;

TORCH_LIBRARY(gs_classes, m) {
  m.class_<Graph>("Graph")
      .def(torch::init<bool>())
      .def("_CAPI_set_data", &Graph::SetData)
      .def("_CAPI_get_data", &Graph::GetData)
      .def("_CAPI_columnwise_slicing", &Graph::ColumnwiseSlicing)
      .def("_CAPI_rowwise_slicing", &Graph::RowwiseSlicing)
      .def("_CAPI_columnwise_sampling", &Graph::ColumnwiseSampling)
      .def("_CAPI_fused_columnwise_slicing_sampling",
           &Graph::ColumnwiseFusedSlicingAndSampling)
      .def("_CAPI_load_csc", &Graph::LoadCSC)
      .def("_CAPI_load_csc_with_col_ids", &Graph::LoadCSCWithColIds)
      .def("_CAPI_row_indices", &Graph::RowIndices)
      .def("_CAPI_print", &Graph::Print)
      .def("_CAPI_all_indices", &Graph::AllIndices)
      .def("_CAPI_relabel", &Graph::Relabel)
      .def("_CAPI_sum", &Graph::Sum)
      .def("_CAPI_l2norm", &Graph::L2Norm)
      .def("_CAPI_divide", &Graph::Divide)
      .def("_CAPI_normalize", &Graph::Normalize)
      .def("_CAPI_metadata", &Graph::MetaData);
}

TORCH_LIBRARY(gs_ops, m) {
  m.def("list_sampling", &ListSampling);
  m.def("list_sampling_with_probs", &ListSamplingProbs);
}

namespace gs {}