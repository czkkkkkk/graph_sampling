#include <torch/custom_class.h>
#include <torch/script.h>

#include "./graph.h"

namespace gs {

TORCH_LIBRARY(graph, m) {
  m.class_<Graph>("Graph")
    .def(torch::init<bool>())
    .def("ColumnwiseSlicing", &Graph::ColumnwiseSlicing)
  ;
}

}