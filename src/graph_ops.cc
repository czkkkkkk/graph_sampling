#include "graph_ops.h"
#include "bcast.h"
#include "cuda/fusion/column_row_slicing.h"
#include "cuda/fusion/node2vec.h"
#include "cuda/fusion/random_walk.h"
#include "cuda/fusion/slice_sampling.h"
#include "cuda/graph_ops.h"
#include "cuda/sddmm.h"
#include "cuda/tensor_ops.h"

namespace gs {

std::shared_ptr<COO> GraphCSC2COO(std::shared_ptr<CSC> csc, bool CSC2COO) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor row, col;
    bool row_sorted = false, col_sorted = false;
    if (CSC2COO) {
      std::tie(row, col) = impl::CSC2COOCUDA(csc->indptr, csc->indices);
      col_sorted = true;
    } else {
      std::tie(col, row) = impl::CSC2COOCUDA(csc->indptr, csc->indices);
      row_sorted = true;
    }
    return std::make_shared<COO>(
        COO{row, col, csc->e_ids, row_sorted, col_sorted});
  } else {
    LOG(FATAL) << "Not implemented warning";
    return std::make_shared<COO>(COO{});
  }
}

std::shared_ptr<CSC> GraphCOO2CSC(std::shared_ptr<COO> coo, int64_t num_items,
                                  bool COO2CSC) {
  if (coo->row.device().type() == torch::kCUDA) {
    torch::Tensor indptr, indices;
    torch::optional<torch::Tensor> sorted_e_ids = torch::nullopt,
                                   sort_index = torch::nullopt;
    if (COO2CSC) {
      std::tie(indptr, indices, sort_index) =
          impl::COO2CSCCUDA(coo->row, coo->col, num_items, coo->col_sorted);
    } else {
      std::tie(indptr, indices, sort_index) =
          impl::COO2CSCCUDA(coo->col, coo->row, num_items, coo->row_sorted);
    }

    if (coo->e_ids.has_value()) {
      if (sort_index.has_value()) {
        sorted_e_ids = coo->e_ids.value().index({sort_index});
      } else {
        sorted_e_ids = coo->e_ids.value();
      }
    } else {
      sorted_e_ids = sort_index;
    }
    return std::make_shared<CSC>(CSC{indptr, indices, sorted_e_ids});
  } else {
    LOG(FATAL) << "Not implemented warning";
    return std::make_shared<CSC>(CSC{});
  }
}

torch::Tensor FusedRandomWalk(std::shared_ptr<CSC> csc, torch::Tensor seeds,
                              int64_t walk_length) {
  torch::Tensor paths = impl::fusion::FusedRandomWalkCUDA(
      seeds, walk_length, csc->indices.data_ptr<int64_t>(),
      csc->indptr.data_ptr<int64_t>());
  return paths;
}

torch::Tensor FusedNode2Vec(std::shared_ptr<CSC> csc, torch::Tensor seeds,
                            int64_t walk_length, double p, double q) {
  torch::Tensor paths = impl::fusion::FusedNode2VecCUDA(
      seeds, walk_length, csc->indices.data_ptr<int64_t>(),
      csc->indptr.data_ptr<int64_t>(), p, q);
  return paths;
}

}  // namespace gs
