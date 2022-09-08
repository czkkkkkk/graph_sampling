#include "graph_ops.h"

#include "cuda_common.h"
#include "utils.h"

namespace gs {
namespace impl {

/*!
 * \brief Given a sorted array and a value this function returns the index
 * of the first element which compares greater than value.
 *
 * This function assumes 0-based index
 * @param A: ascending sorted array
 * @param n: size of the A
 * @param x: value to search in A
 * @return index, i, of the first element st. A[i]>x. If x>=A[n-1] returns n.
 * if x<A[0] then it returns 0.
 */
template <typename IdType>
__device__ IdType _UpperBound(const IdType* A, int64_t n, IdType x) {
  IdType l = 0, r = n, m = 0;
  while (l < r) {
    m = l + (r - l) / 2;
    if (x >= A[m]) {
      l = m + 1;
    } else {
      r = m;
    }
  }
  return l;
}

/*!
 * \brief Repeat elements
 * \param val Value to repeat
 * \param repeats Number of repeats for each value
 * \param pos The position of the output buffer to write the value.
 * \param out Output buffer.
 * \param length Number of values
 *
 * For example:
 * val = [3, 0, 1]
 * repeats = [1, 0, 2]
 * pos = [0, 1, 1]  # write to output buffer position 0, 1, 1
 * then,
 * out = [3, 1, 1]
 */
template <typename IdType, typename DType>
__global__ void _RepeatKernel(const DType* val, const IdType* pos, DType* out,
                              int64_t n_col, int64_t length) {
  IdType tx = static_cast<IdType>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    IdType i = _UpperBound(pos, n_col, tx) - 1;
    out[tx] = val[i];
    tx += stride_x;
  }
}

std::pair<torch::Tensor, torch::Tensor> GraphCSC2COOCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor column_ids) {
  auto coo_size = indices.numel();
  auto col = torch::zeros(coo_size, column_ids.options());

  dim3 block(128);
  dim3 grid((coo_size + block.x - 1) / block.x);
  _RepeatKernel<int64_t, int64_t><<<grid, block>>>(
      column_ids.data_ptr<int64_t>(), indptr.data_ptr<int64_t>(),
      col.data_ptr<int64_t>(), indptr.numel(), coo_size);

  return {indices, col};
}

template <typename IdType>
inline std::vector<torch::Tensor> coo_sort(torch::Tensor coo_key,
                                           torch::Tensor coo_value,
                                           bool need_index) {
  int num_items = coo_key.numel();

  torch::Tensor input_key = coo_key;
  torch::Tensor input_value;
  torch::Tensor output_key = torch::zeros_like(coo_key);
  torch::Tensor output_value;

  if (need_index) {
    input_value = torch::arange(
        num_items, torch::dtype(torch::kInt64).device(torch::kCUDA));
  } else {
    input_value = coo_value;
  }
  output_value = torch::zeros_like(input_value);

  DATA_TYPE_SWITCH(input_value.dtype(), VType, {
    cub::DoubleBuffer<IdType> d_keys(input_key.data_ptr<IdType>(),
                                     output_key.data_ptr<IdType>());
    cub::DoubleBuffer<VType> d_values(input_value.data_ptr<VType>(),
                                      output_value.data_ptr<VType>());
    cub_sortPairs<IdType, VType>(d_keys, d_values, num_items);
  });

  if (need_index) {
    return {output_key, coo_value.index({output_value}), output_value};
  } else {
    return {output_key, output_value};
  }
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
 * needle = [0, 1, 2, 3]
 * then,
 * out = [2, 3, 5, 5]
 */
template <typename IdType>
__global__ void _SortedSearchKernelUpperBound(const IdType* hay,
                                              int64_t hay_size,
                                              const IdType* needles,
                                              int64_t num_needles,
                                              IdType* pos) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride_x = gridDim.x * blockDim.x;
  while (tx < num_needles) {
    const IdType ele = needles[tx];
    // binary search
    IdType lo = 0, hi = hay_size;
    while (lo < hi) {
      IdType mid = (lo + hi) >> 1;
      if (hay[mid] <= ele) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    pos[tx] = lo;
    tx += stride_x;
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GraphCOO2CSRCUDA(
    torch::Tensor row, torch::Tensor col) {
  torch::Tensor sort_row, sort_col;
  auto sort_results = coo_sort<int64_t>(row, col, false);
  sort_row = sort_results[0];
  sort_col = sort_results[1];

  auto row_ids = TensorUniqueCUDA(sort_row);
  auto row_size = row_ids.numel();
  auto indptr = torch::zeros(row_size + 1, row_ids.options());

  dim3 block(128);
  dim3 grid((row_size + block.x - 1) / block.x);
  _SortedSearchKernelUpperBound<int64_t><<<grid, block>>>(
      sort_row.data_ptr<int64_t>(), sort_row.numel(),
      row_ids.data_ptr<int64_t>(), row_size, indptr.data_ptr<int64_t>() + 1);

  return std::make_tuple(row_ids, indptr, sort_col);
}

}  // namespace impl
}  // namespace gs
