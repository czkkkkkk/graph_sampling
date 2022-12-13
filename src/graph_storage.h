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

struct _TMP {
  torch::Tensor indptr;
  torch::Tensor coo_in_indptr;
  torch::Tensor coo_in_indices;
};

#define _CSR 4
#define _CSC 2
#define _COO 1

}  // namespace gs

#endif