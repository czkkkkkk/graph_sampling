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
  torch::Tensor data;
};

}  // namespace gs

#endif