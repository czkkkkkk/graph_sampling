#include "graph_ops.h"

#include "cuda_common.h"
#include "utils.h"

namespace gs {
namespace impl {

template <typename IdType>
torch::Tensor GetSubIndptr(torch::Tensor indptr, torch::Tensor column_ids) {
  int64_t size = column_ids.numel();
  auto new_indptr = torch::zeros(size + 1, torch::dtype(indptr.dtype()).device(torch::kCUDA));

  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(
      thrust::device, it(0), it(size),
      [in = column_ids.data_ptr<IdType>(),
       in_indptr = indptr.data_ptr<IdType>(),
       out = new_indptr.data_ptr<IdType>()] __device__(int i) mutable {
        IdType begin = in_indptr[in[i]];
        IdType end = in_indptr[in[i] + 1];
        out[i] = end - begin;
      });

  cub_exclusiveSum<IdType>(new_indptr.data_ptr<IdType>(), size + 1);
  return new_indptr;
}

template <typename IdType>
__global__ void _GetSubIndicesKernel(IdType* out_indices, IdType* select_index,
                                     IdType* indptr, IdType* indices,
                                     IdType* sub_indptr, IdType* column_ids,
                                     int64_t size) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;

  while (row < size) {
    IdType in_start = indptr[column_ids[row]];
    IdType out_start = sub_indptr[row];
    IdType n_edges = sub_indptr[row + 1] - sub_indptr[row];
    IdType in_pos, out_pos;
    for (int idx = threadIdx.x; idx < n_edges; idx += blockDim.x) {
      out_pos = out_start + idx;
      in_pos = in_start + idx;
      out_indices[out_pos] = indices[in_pos];
      select_index[out_pos] = in_pos;
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType>
std::pair<torch::Tensor, torch::Tensor> GetSubIndices(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor sub_indptr,
    torch::Tensor column_ids) {
  int64_t size = column_ids.numel();
  thrust::device_ptr<IdType> item_prefix(
      static_cast<IdType*>(sub_indptr.data_ptr<IdType>()));
  int n_edges = item_prefix[size];  // cpu
  auto sub_indices = torch::zeros(n_edges, torch::dtype(indices.dtype()).device(torch::kCUDA));
  auto select_index = torch::zeros(n_edges, torch::dtype(indices.dtype()).device(torch::kCUDA));

  dim3 block(32, 16);
  dim3 grid((size + block.x - 1) / block.x);
  _GetSubIndicesKernel<int64_t><<<grid, block>>>(
      sub_indices.data_ptr<int64_t>(), select_index.data_ptr<int64_t>(),
      indptr.data_ptr<int64_t>(), indices.data_ptr<int64_t>(),
      sub_indptr.data_ptr<int64_t>(), column_ids.data_ptr<int64_t>(), size);
  return {sub_indices, select_index};
}

// columwise slicing
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> OnIndptrSlicingCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor column_ids) {
  torch::Tensor sub_indptr, sub_indices, select_index;
  sub_indptr = GetSubIndptr<int64_t>(indptr, column_ids);
  std::tie(sub_indices, select_index) =
      GetSubIndices<int64_t>(indptr, indices, sub_indptr, column_ids);
  return {sub_indptr, sub_indices, select_index};
}

}  // namespace impl
}  // namespace gs