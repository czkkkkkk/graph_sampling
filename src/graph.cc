#include "./graph.h"

#include <sstream>
#include "bcast.h"
#include "cuda/fusion/column_row_slicing.h"
#include "cuda/fusion/edge_map_reduce.h"
#include "cuda/graph_ops.h"
#include "cuda/sddmm.h"
#include "cuda/tensor_ops.h"
#include "graph_ops.h"

namespace gs {

void Graph::CreateSparseFormat(int64_t format) {
  if (format == _COO) {
    if (coo_ != nullptr) return;
    if (csc_ != nullptr) {
      SetCOO(GraphCSC2COO(csc_, true));
    } else {
      SetCOO(GraphCSC2COO(csr_, false));
    }
  } else if (format == _CSC) {
    if (csc_ != nullptr) return;
    if (coo_ != nullptr)
      SetCSC(GraphCOO2CSC(coo_, num_cols_, true));
    else {
      SetCOO(GraphCSC2COO(csr_, false));
      SetCSC(GraphCOO2CSC(coo_, num_cols_, true));
    }
  } else if (format == _CSR) {
    if (csr_ != nullptr) return;
    if (coo_ != nullptr)
      SetCSR(GraphCOO2CSC(coo_, num_rows_, false));
    else {
      SetCOO(GraphCSC2COO(csc_, true));
      SetCSR(GraphCOO2CSC(coo_, num_rows_, false));
    }
  } else {
    LOG(FATAL) << "Unsupported sparse format!";
  }
}

torch::Tensor Graph::GetCSCIndptr() {
  CreateSparseFormat(_CSC);
  return csc_->indptr;
}
torch::Tensor Graph::GetCSCIndices() {
  CreateSparseFormat(_CSC);
  return csc_->indices;
}
torch::Tensor Graph::GetCSCEids() {
  CreateSparseFormat(_CSC);
  return csc_->e_ids.has_value() ? csc_->e_ids.value() : torch::Tensor();
}
torch::Tensor Graph::GetCOORows() {
  CreateSparseFormat(_COO);
  return coo_->row;
}
torch::Tensor Graph::GetCOOCols() {
  CreateSparseFormat(_COO);
  return coo_->col;
}
torch::Tensor Graph::GetCOOEids() {
  CreateSparseFormat(_COO);
  return coo_->e_ids.has_value() ? coo_->e_ids.value() : torch::Tensor();
}
torch::Tensor Graph::GetCSRIndptr() {
  CreateSparseFormat(_CSR);
  return csr_->indptr;
}
torch::Tensor Graph::GetCSRIndices() {
  CreateSparseFormat(_CSR);
  return csr_->indices;
}
torch::Tensor Graph::GetCSREids() {
  CreateSparseFormat(_CSR);
  return csr_->e_ids.has_value() ? csr_->e_ids.value() : torch::Tensor();
}

c10::intrusive_ptr<Graph> Graph::Slicing(torch::Tensor seeds, int64_t axis,
                                         int64_t on_format,
                                         int64_t output_format, bool compact) {
  auto ret = c10::intrusive_ptr<Graph>(
      std::unique_ptr<Graph>(new Graph(this->num_rows_, this->num_cols_)));
  return ret;
}

torch::Tensor Graph::RandomWalk(torch::Tensor seeds, int64_t walk_length) {
  return FusedRandomWalk(this->csc_, seeds, walk_length);
}

}  // namespace gs
