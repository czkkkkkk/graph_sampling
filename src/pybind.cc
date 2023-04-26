#include <torch/custom_class.h>
#include <torch/script.h>

#include "./graph.h"
#include "./graph_ops.h"
#include "./tensor_ops.h"
#include "cuda/tensor_ops.h"
using namespace gs;

TORCH_LIBRARY(gs_classes, m) {
  m.class_<Graph>("Graph")
      .def(torch::init<int64_t, int64_t>())
      .def("_CAPI_LoadCSC", &Graph::LoadCSC)
      .def("_CAPI_LoadCOO", &Graph::LoadCOO)
      .def("_CAPI_LoadCSR", &Graph::LoadCSR)
      .def("_CAPI_GetNumRows", &Graph::GetNumRows)
      .def("_CAPI_GetNumCols", &Graph::GetNumCols)
      .def("_CAPI_GetNumEdges", &Graph::GetNumEdges)
      .def("_CAPI_GetCSCIndptr", &Graph::GetCSCIndptr)
      .def("_CAPI_GetCSCIndices", &Graph::GetCSCIndices)
      .def("_CAPI_GetCSCEdges", &Graph::GetCSCEids)
      .def("_CAPI_GetCOORows", &Graph::GetCOORows)
      .def("_CAPI_GetCooCols", &Graph::GetCOOCols)
      .def("_CAPI_GetCOOEids", &Graph::GetCOOEids)
      .def("_CAPI_GetCSRIndptr", &Graph::GetCSRIndptr)
      .def("_CAPI_GetCSRIndices", &Graph::GetCSRIndices)
      .def("_CAPI_GetCSREids", &Graph::GetCSREids)
      .def("_CAPI_Slicing", &Graph::Slicing)
      .def("_CAPI_RandomWalk", &Graph::RandomWalk)
      .def("_CAPI_Node2Vec", &Graph::Node2Vec);
}

TORCH_LIBRARY(gs_ops, m) {
  m.def("_CAPI_ListSampling", &ListSampling);
  m.def("_CAPI_ListSamplingWithProbs", &ListSamplingProbs);
}

namespace gs {}