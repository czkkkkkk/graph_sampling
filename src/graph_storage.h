#ifndef GS_GRAPH_STORAGE_H_
#define GS_GRAPH_STORAGE_H_

#include "./logging.h"
#include "torch/custom_class.h"

#define assertm(exp, msg) assert(((void)msg, exp))

namespace gs {

struct CSC {
  torch::Tensor col_ids;
  torch::Tensor indptr;
  torch::Tensor indices;
};

struct CSR {
  torch::Tensor row_ids;
  torch::Tensor indptr;
  torch::Tensor indices;
};

struct COO {
  torch::Tensor row;
  torch::Tensor col;
};

}  // namespace gs

#endif