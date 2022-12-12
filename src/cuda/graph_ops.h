#ifndef GS_CUDA_GRAPH_OPS_H_
#define GS_CUDA_GRAPH_OPS_H_

#include <torch/torch.h>
#include "./logging.h"

namespace gs {
namespace impl {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CSCColSlicingCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor column_ids);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> OnIndptrSlicingCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor nid_map,
    torch::Tensor column_ids);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> OnIndicesSlicingCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor row_ids);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> COORowSlicingCUDA(
    torch::Tensor coo_row, torch::Tensor coo_col, torch::Tensor row_ids);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CSCColSamplingCUDA(
    torch::Tensor indptr, torch::Tensor indices, int64_t fanout, bool replace);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CSCColSamplingProbsCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor probs,
    int64_t fanout, bool replace);

void CSCSumCUDA(torch::Tensor indptr, torch::optional<torch::Tensor> e_ids,
                torch::optional<torch::Tensor> n_ids, torch::Tensor data,
                torch::Tensor out_data, int64_t powk);

void COOSumCUDA(torch::Tensor target, torch::optional<torch::Tensor> e_ids,
                torch::Tensor data, torch::Tensor out_data, int64_t powk);

torch::Tensor CSCDivCUDA(torch::Tensor indptr,
                         torch::optional<torch::Tensor> e_ids,
                         torch::optional<torch::Tensor> data,
                         torch::Tensor divisor);

std::pair<torch::Tensor, torch::Tensor> CSC2COOCUDA(torch::Tensor indptr,
                                                    torch::Tensor indices);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> COO2CSCCUDA(
    torch::Tensor row, torch::Tensor col, int64_t num_rows);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> COO2DCSCCUDA(
    torch::Tensor row, torch::Tensor col, torch::Tensor ids);

std::pair<torch::Tensor, torch::Tensor> DCSC2COOCUDA(torch::Tensor indptr,
                                                     torch::Tensor indices,
                                                     torch::Tensor ids);

}  // namespace impl
}  // namespace gs

#endif