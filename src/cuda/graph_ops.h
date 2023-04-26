#ifndef GS_CUDA_GRAPH_OPS_H_
#define GS_CUDA_GRAPH_OPS_H_

#include <torch/torch.h>
#include "./logging.h"

namespace gs {
namespace impl {

// slicing

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
OnIndptrSlicingCUDA(torch::Tensor indptr, torch::Tensor indices,
                    torch::Tensor seeds, bool with_coo);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
OnIndicesSlicingCUDA(torch::Tensor indptr, torch::Tensor indices,
                     torch::Tensor row_ids, bool with_coo);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> COORowSlicingCUDA(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor row_ids);

std::pair<torch::Tensor, torch::Tensor> CSC2COOCUDA(torch::Tensor indptr,
                                                    torch::Tensor indices);

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
COO2CSCCUDA(torch::Tensor row, torch::Tensor col, int64_t num_cols,
            bool col_sorted);

}  // namespace impl
}  // namespace gs

#endif