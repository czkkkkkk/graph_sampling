#include "graph_ops.h"

#include "cuda_common.h"
#include "utils.h"

namespace gs {
namespace impl {

template <typename IdType>
torch::Tensor GetSubIndptr(torch::Tensor indptr, torch::Tensor column_ids) {
  int64_t size = column_ids.numel();
  auto new_indptr = torch::zeros(size + 1, indptr.options());
  thrust::device_ptr<IdType> item_prefix(
      static_cast<IdType*>(new_indptr.data_ptr<IdType>()));

  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(
      thrust::device, it(0), it(size),
      [in = column_ids.data_ptr<IdType>(),
       in_indptr = indptr.data_ptr<IdType>(),
       out = thrust::raw_pointer_cast(item_prefix)] __device__(int i) mutable {
        IdType begin = in_indptr[in[i]];
        IdType end = in_indptr[in[i] + 1];
        out[i] = end - begin;
      });

  cub_exclusiveSum<IdType>(thrust::raw_pointer_cast(item_prefix), size + 1);
  return new_indptr;
}

template <typename IdType>
__global__ void _GetSubIndicesKernel(IdType* out_indices, IdType* indptr,
                                     IdType* indices, IdType* sub_indptr,
                                     IdType* column_ids, int64_t size) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;

  while (row < size) {
    int64_t in_start = indptr[column_ids[row]];
    int64_t out_start = sub_indptr[row];
    int64_t n_edges = sub_indptr[row + 1] - sub_indptr[row];
    for (int idx = threadIdx.x; idx < n_edges; idx += blockDim.x) {
      out_indices[out_start + idx] = indices[in_start + idx];
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType>
torch::Tensor GetSubIndices(torch::Tensor indptr, torch::Tensor indices,
                            torch::Tensor sub_indptr,
                            torch::Tensor column_ids) {
  int64_t size = column_ids.numel();
  thrust::device_ptr<IdType> item_prefix(
      static_cast<IdType*>(sub_indptr.data_ptr<IdType>()));
  int n_edges = item_prefix[size];  // cpu
  auto sub_indices = torch::zeros(n_edges, indices.options());

  dim3 block(32, 16);
  dim3 grid((size + block.x - 1) / block.x);
  _GetSubIndicesKernel<int64_t><<<grid, block>>>(
      sub_indices.data_ptr<int64_t>(), indptr.data_ptr<int64_t>(),
      indices.data_ptr<int64_t>(), sub_indptr.data_ptr<int64_t>(),
      column_ids.data_ptr<int64_t>(), size);
  return sub_indices;
}

// columwise slicing
std::pair<torch::Tensor, torch::Tensor> CSCColumnwiseSlicingCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor column_ids) {
  auto sub_indptr = GetSubIndptr<int64_t>(indptr, column_ids);
  auto sub_indices =
      GetSubIndices<int64_t>(indptr, indices, sub_indptr, column_ids);
  return {sub_indptr, sub_indices};
}

std::pair<torch::Tensor, torch::Tensor> NormalizeCUDA(torch::Tensor indptr,
                                                      torch::Tensor indices) {
  int64_t size = indptr.numel() - 1;
  // auto ind_sum = 
}

}  // namespace impl
}  // namespace gs