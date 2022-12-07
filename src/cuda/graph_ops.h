#ifndef GS_CUDA_GRAPH_OPS_H_
#define GS_CUDA_GRAPH_OPS_H_

#include <torch/torch.h>
#include "./logging.h"

namespace gs {
namespace impl {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> OnIndptrSlicingCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor column_ids);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> OnIndicesSlicingCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor row_ids);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CSCColSamplingCUDA(
    torch::Tensor indptr, torch::Tensor indices, int64_t fanout, bool replace);

torch::Tensor CSCSumCUDA(torch::Tensor indptr,
                         torch::optional<torch::Tensor> e_ids,
                         torch::optional<torch::Tensor> data, int64_t powk);

torch::Tensor CSCDivCUDA(torch::Tensor indptr,
                         torch::optional<torch::Tensor> e_ids,
                         torch::optional<torch::Tensor> data,
                         torch::Tensor divisor);

torch::Tensor CSCNormalizeCUDA(torch::Tensor indptr,
                               torch::optional<torch::Tensor> e_ids,
                               torch::optional<torch::Tensor> data);

std::pair<torch::Tensor, torch::Tensor> CSC2COOCUDA(torch::Tensor indptr,
                                                    torch::Tensor indices);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> COO2CSCCUDA(
    torch::Tensor row, torch::Tensor col, int64_t num_rows);

}  // namespace impl
}  // namespace gs

#endif