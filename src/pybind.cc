#include <torch/custom_class.h>
#include <torch/script.h>

#include "./graph.h"

using namespace gs;

TORCH_LIBRARY(gs_classes, m) {
  m.class_<Graph>("Graph")
      .def(torch::init<bool>())
      .def("columnwise_slicing", &Graph::ColumnwiseSlicing)
      .def("columnwise_sampling", &Graph::ColumnwiseSampling)
      .def("fused_columnwise_slicing_sampling", &Graph::ColumnwiseFusedSlicingAndSampling)
      .def("load_csc", &Graph::LoadCSC)
      .def("load_csc_with_col_ids", &Graph::LoadCSCWithColIds)
      .def("row_indices", &Graph::RowIndices)
      .def("print", &Graph::Print)
      .def("all_indices",&Graph::AllIndices)
      .def("relabel",&Graph::Relabel)
      .def("_CAPI_metadata",&Graph::MetaData);
}

namespace gs {}