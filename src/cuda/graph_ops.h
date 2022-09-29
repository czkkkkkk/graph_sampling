#ifndef GS_CUDA_GRAPH_OPS_H_
#define GS_CUDA_GRAPH_OPS_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include "./logging.h"

namespace gs {
namespace impl {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> OnIndptrSlicingCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor column_ids);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> OnIndicesSlicingCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor row_ids);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
CSCColumnwiseSamplingCUDA(torch::Tensor indptr, torch::Tensor indices,
                          int64_t fanout, bool replace);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
CSCColumnwiseFusedSlicingAndSamplingCUDA(torch::Tensor indptr,
                                         torch::Tensor indices,
                                         torch::Tensor column_ids,
                                         int64_t fanout, bool replace);

torch::Tensor TensorUniqueCUDA(torch::Tensor input);

std::tuple<torch::Tensor, std::vector<torch::Tensor>> RelabelCUDA2(
    std::vector<torch::Tensor> mapping_tensor,
    std::vector<torch::Tensor> data_requiring_relabel);

torch::Tensor GraphSumCUDA(torch::Tensor indptr,
                           torch::optional<torch::Tensor> e_ids,
                           torch::optional<torch::Tensor> data, int64_t powk);

torch::Tensor GraphL2NormCUDA(torch::Tensor indptr,
                              torch::optional<torch::Tensor> e_ids,
                              torch::optional<torch::Tensor> data);

torch::Tensor GraphDivCUDA(torch::Tensor indptr,
                           torch::optional<torch::Tensor> e_ids,
                           torch::optional<torch::Tensor> data,
                           torch::Tensor divisor);

torch::Tensor GraphNormalizeCUDA(torch::Tensor indptr,
                                 torch::optional<torch::Tensor> e_ids,
                                 torch::optional<torch::Tensor> data);

std::pair<torch::Tensor, torch::Tensor> GraphCSC2COOCUDA(torch::Tensor indptr,
                                                         torch::Tensor indices);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GraphCOO2CSRCUDA(
    torch::Tensor row, torch::Tensor col, int64_t num_rows);

}  // namespace impl
}  // namespace gs

#endif