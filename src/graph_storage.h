#ifndef GS_GRAPH_STORAGE_H_
#define GS_GRAPH_STORAGE_H_

#include "torch/custom_class.h"

namespace gs {

struct CSC {
  torch::Tensor indptr, indices;
};

}  // namespace gs

#endif