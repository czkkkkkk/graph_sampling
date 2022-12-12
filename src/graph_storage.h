#ifndef GS_GRAPH_STORAGE_H_
#define GS_GRAPH_STORAGE_H_

#include "./logging.h"
#include "torch/custom_class.h"

namespace gs {

struct CSC {
  torch::Tensor indptr;
  torch::Tensor indices;
  torch::optional<torch::Tensor> e_ids;
};

typedef CSC CSR;

struct COO {
  torch::Tensor row;
  torch::Tensor col;
  torch::optional<torch::Tensor> e_ids;
};

#define _CSR 0
#define _CSC 1
#define _COO 2

}  // namespace gs

#endif