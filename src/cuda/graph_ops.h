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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
CSCColumnwiseSamplingCUDA(torch::Tensor indptr, torch::Tensor indices,
                          int64_t fanout, bool replace);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
CSCColumnwiseFusedSlicingAndSamplingCUDA(torch::Tensor indptr,
                                         torch::Tensor indices,
                                         torch::Tensor column_ids,
                                         int64_t fanout, bool replace);

torch::Tensor RandomWalkFusedCUDA(torch::Tensor seeds, int64_t walklength,
                                  int64_t* indices, int64_t* indptr);
torch::Tensor TensorUniqueCUDA(torch::Tensor input);

// RelabelCUDA leverages vector<Tensor> mapping_tensor to create the hashmap
// which stores the mapping. Then, it will do relabel operation for tensor in
// data_requiring_relabel with the hashmap.
// It return {unique_tensor, {tensor1_after_relabeled,
// tensor2_after_relabeled, ...}}.
std::tuple<torch::Tensor, std::vector<torch::Tensor>> RelabelCUDA(
    std::vector<torch::Tensor> mapping_tensor,
    std::vector<torch::Tensor> data_requiring_relabel);

torch::Tensor GraphSumCUDA(torch::Tensor indptr,
                           torch::optional<torch::Tensor> e_ids,
                           torch::optional<torch::Tensor> data, int64_t powk);

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