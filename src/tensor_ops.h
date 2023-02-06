#ifndef GS_TENSOR_OPS_H_
#define GS_TENSOR_OPS_H_

#include <torch/script.h>

namespace gs {

/**
 * @brief ListSampling, using A-Res sampling for replace = False and uniform
 * sampling for replace = True. Tt will return (selected_data, selected_index)
 *
 * @param data
 * @param num_picks
 * @param replace
 * @return std::tuple<torch::Tensor, torch::Tensor>
 */
std::tuple<torch::Tensor, torch::Tensor> ListSampling(torch::Tensor data,
                                                      int64_t num_picks,
                                                      bool replace);

std::tuple<torch::Tensor, torch::Tensor> ListSamplingProbs(torch::Tensor data,
                                                           torch::Tensor probs,
                                                           int64_t num_picks,
                                                           bool replace);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> BatchListSamplingProbs(
    torch::Tensor data, torch::Tensor probs, int64_t num_picks, bool replace,
    torch::Tensor range);

torch::Tensor IndexSearch(torch::Tensor origin_data, torch::Tensor keys);

std::tuple<torch::Tensor, torch::Tensor> BatchUnique(
    const std::vector<torch::Tensor> &batch_tensors,
    const std::vector<torch::Tensor> &segment_ptrs, int64_t num_batchs);

std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>>
BatchRelabel(const std::vector<torch::Tensor> &batch_tensors,
             const std::vector<torch::Tensor> &segment_ptrs,
             int64_t num_batchs);
std::vector<torch::Tensor> SplitByOffset(torch::Tensor data,
                                         torch::Tensor offset);
}  // namespace gs
#endif