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
      .def("row_indices", &Graph::RowIndices)
      .def("print", &Graph::Print);
  ;
}

namespace gs {}