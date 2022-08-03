#include <torch/custom_class.h>
#include <torch/script.h>

#include "./graph.h"

using namespace gs;

TORCH_LIBRARY(gs_classes, m) {
  m.class_<Graph>("Graph")
    .def(torch::init<bool>())
    .def("columnwise_slicing", &Graph::ColumnwiseSlicing)
  ;
}

namespace gs {

}