#include "graph_ops.h"

#include "cuda_common.h"
#include "utils.h"

namespace gs {
namespace impl {

/*!
 * @brief Repeat elements.
 *
 * @param pos: The position of the output buffer to write the value.
 * @param out: Output buffer.
 * @param n_col: Length of positions
 * @param length: Number of values
 *
 * For example:
 * pos = [0, 1, 3, 4]
 * (implicit) val = [0, 1, 2]
 * then,
 * out = [0, 1, 1, 2]
 */
template <typename IdType>
__global__ void _RepeatKernel(const IdType* pos, IdType* out, int64_t n_col,
                              int64_t length) {
  IdType tx = static_cast<IdType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    IdType i = cub::UpperBound(pos, n_col, tx) - 1;
    out[tx] = i;
    tx += stride_x;
  }
}

std::pair<torch::Tensor, torch::Tensor> CSC2COOCUDA(
    torch::Tensor indptr, torch::Tensor indices) {
  auto coo_size = indices.numel();
  auto col = torch::zeros(coo_size, indptr.options());

  dim3 block(128);
  dim3 grid((coo_size + block.x - 1) / block.x);
  _RepeatKernel<int64_t><<<grid, block>>>(indptr.data_ptr<int64_t>(),
                                          col.data_ptr<int64_t>(),
                                          indptr.numel(), coo_size);
  return {indices, col};
}

template <typename IdType>
inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> COOSort(
    torch::Tensor coo_key, torch::Tensor coo_value) {
  int num_items = coo_key.numel();

  torch::Tensor input_key = coo_key;
  torch::Tensor input_value;
  torch::Tensor output_key = torch::zeros_like(coo_key);
  torch::Tensor output_value;

  input_value = torch::arange(num_items,
                              torch::dtype(torch::kInt64).device(torch::kCUDA));
  output_value = torch::zeros_like(input_value);
  cub_sortPairs<IdType, int64_t>(input_key.data_ptr<IdType>(),
                                 output_key.data_ptr<IdType>(),
                                 input_value.data_ptr<int64_t>(),
                                 output_value.data_ptr<int64_t>(), num_items);

  return {output_key, coo_value.index({output_value}), output_value};
}

/*!
 * \brief Search for the insertion positions for needle in the hay.
 *
 * The hay is a list of sorted elements and the result is the insertion position
 * of each needle so that the insertion still gives sorted order.
 *
 * It essentially perform binary search to find upper bound for each needle
 * elements.
 *
 * For example:
 * hay = [0, 0, 1, 2, 2]
 * (implicit) needle = [0, 1, 2, 3]
 * then,
 * out = [2, 3, 5, 5]
 *
 * hay = [0, 0, 1, 3, 3]
 * (implicit) needle = [0, 1, 2, 3]
 * then,
 * out = [2, 3, 3, 5]
 */
template <typename IdType>
__global__ void _SortedSearchKernelUpperBound(const IdType* hay,
                                              int64_t hay_size,
                                              int64_t num_needles,
                                              IdType* pos) {
  IdType tx = static_cast<IdType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < num_needles) {
    pos[tx] = cub::UpperBound(hay, hay_size, tx);
    tx += stride_x;
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> COO2CSCCUDA(
    torch::Tensor row, torch::Tensor col, int64_t num_rows) {
  torch::Tensor sort_row, sort_col, sort_index;
  std::tie(sort_row, sort_col, sort_index) = COOSort<int64_t>(col, row);

  auto row_size = num_rows;
  auto indptr = torch::zeros(row_size + 1, sort_row.options());

  dim3 block(128);
  dim3 grid((row_size + block.x - 1) / block.x);
  _SortedSearchKernelUpperBound<int64_t>
      <<<grid, block>>>(sort_row.data_ptr<int64_t>(), sort_row.numel(),
                        row_size, indptr.data_ptr<int64_t>() + 1);
  return std::make_tuple(indptr, sort_col, sort_index);
}

}  // namespace impl
}  // namespace gs
