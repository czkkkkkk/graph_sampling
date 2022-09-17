#include "graph_ops.h"

#include "cuda_common.h"
#include "utils.h"

namespace gs {
namespace impl {

template <typename IdType>
torch::Tensor GetSubIndptr(torch::Tensor indptr, torch::Tensor column_ids) {
  int64_t size = column_ids.numel();
  auto new_indptr = torch::zeros(size + 1, indptr.options());

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
__global__ void _GetSubIndicesWithEIDKernel(IdType* out_indices,
                                            IdType* out_e_ids, IdType* indptr,
                                            IdType* indices, IdType* e_ids,
                                            IdType* sub_indptr,
                                            IdType* column_ids, int64_t size) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;

  while (row < size) {
    IdType in_start = indptr[column_ids[row]];
    IdType out_start = sub_indptr[row];
    IdType n_edges = sub_indptr[row + 1] - sub_indptr[row];
    for (int idx = threadIdx.x; idx < n_edges; idx += blockDim.x) {
      out_indices[out_start + idx] = indices[in_start + idx];
      out_e_ids[out_start + idx] = e_ids[in_start + idx];
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType>
std::pair<torch::Tensor, torch::Tensor> GetSubIndicesWithEID(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor e_ids,
    torch::Tensor sub_indptr, torch::Tensor column_ids) {
  int64_t size = column_ids.numel();
  thrust::device_ptr<IdType> item_prefix(
      static_cast<IdType*>(sub_indptr.data_ptr<IdType>()));
  int n_edges = item_prefix[size];  // cpu
  auto sub_indices = torch::zeros(n_edges, indices.options());
  auto sub_e_ids = torch::zeros(n_edges, e_ids.options());

  dim3 block(32, 16);
  dim3 grid((size + block.x - 1) / block.x);
  _GetSubIndicesWithEIDKernel<int64_t><<<grid, block>>>(
      sub_indices.data_ptr<int64_t>(), sub_e_ids.data_ptr<int64_t>(),
      indptr.data_ptr<int64_t>(), indices.data_ptr<int64_t>(),
      e_ids.data_ptr<int64_t>(), sub_indptr.data_ptr<int64_t>(),
      column_ids.data_ptr<int64_t>(), size);
  return {sub_indices, sub_e_ids};
}

// columwise slicing
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CSCColumnwiseSlicingCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor e_ids,
    torch::Tensor column_ids) {
  torch::Tensor sub_indptr, sub_indices, sub_e_ids;
  sub_indptr = GetSubIndptr<int64_t>(indptr, column_ids);
  std::tie(sub_indices, sub_e_ids) = GetSubIndicesWithEID<int64_t>(
      indptr, indices, e_ids, sub_indptr, column_ids);
  return {sub_indptr, sub_indices, sub_e_ids};
}

}  // namespace impl
}  // namespace gs