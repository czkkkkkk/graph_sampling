#ifndef GS_GRAPH_STORAGE_H_
#define GS_GRAPH_STORAGE_H_

#include "./logging.h"
#include "torch/custom_class.h"

#define assertm(exp, msg) assert(((void)msg, exp))

namespace gs {

struct CSC {
  torch::Tensor indptr;
  torch::Tensor indices;
  torch::optional<torch::Tensor> e_ids;
};

struct CSR {
  torch::Tensor indptr;
  torch::Tensor indices;
  torch::optional<torch::Tensor> e_ids;
};

struct COO {
  torch::Tensor row;
  torch::Tensor col;
  torch::optional<torch::Tensor> e_ids;
};

}  // namespace gs

#endif