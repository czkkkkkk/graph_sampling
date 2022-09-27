#include "graph_ops.h"

#include "atomic.h"
#include "cuda_common.h"
#include "utils.h"

namespace gs {
namespace impl {
template <typename IdType, typename DType>
__global__ void GroupSum(IdType* indptr, IdType* e_ids, DType* data,
                         DType* out_data, int64_t size) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;

  while (row < size) {
    IdType start = indptr[row];
    IdType n_edges = indptr[row + 1] - start;
    for (int idx = threadIdx.x; idx < n_edges; idx += blockDim.x) {
      IdType pos = (e_ids == nullptr) ? start + idx : e_ids[start + idx];
      AtomicAdd(&out_data[row], (data == nullptr) ? 1.0 : data[start + idx]);
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType, typename DType>
__global__ void GroupNormL2(IdType* indptr, IdType* e_ids, DType* data,
                            DType* out_data, int64_t size) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;

  while (row < size) {
    IdType start = indptr[row];
    IdType n_edges = indptr[row + 1] - start;
    for (int idx = threadIdx.x; idx < n_edges; idx += blockDim.x) {
      IdType pos = (e_ids == nullptr) ? start + idx : e_ids[start + idx];
      AtomicAdd(&out_data[row], (data == nullptr) ? 1.0 : powf(data[pos], 2));
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType, typename DType>
__global__ void GroupDiv(IdType* indptr, IdType* e_ids, DType* data,
                         DType* divisor, DType* out_data, int64_t size) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;

  while (row < size) {
    IdType start = indptr[row];
    IdType n_edges = indptr[row + 1] - start;
    for (int idx = threadIdx.x; idx < n_edges; idx += blockDim.x) {
      IdType pos = (e_ids == nullptr) ? start + idx : e_ids[start + idx];
      out_data[pos] = ((data == nullptr) ? 1.0 : data[pos]) / divisor[row];
    }
    row += gridDim.x * blockDim.y;
  }
}

torch::Tensor GraphSumCUDA(torch::Tensor indptr,
                           torch::optional<torch::Tensor> e_ids,
                           torch::optional<torch::Tensor> data) {
  auto size = indptr.numel() - 1;
  auto data_ptr =
      (data.has_value()) ? data.value().data_ptr<_Float32>() : nullptr;
  auto e_ids_ptr =
      (e_ids.has_value()) ? e_ids.value().data_ptr<int64_t>() : nullptr;
  auto options = (data.has_value())
                     ? data.value().options()
                     : torch::dtype(torch::kFloat32).device(torch::kCUDA);
  auto group_sum = torch::zeros(size, options);

  dim3 block(32, 16);
  dim3 grid((size + block.x - 1) / block.x);
  GroupSum<int64_t, _Float32>
      <<<grid, block>>>(indptr.data_ptr<int64_t>(), e_ids_ptr, data_ptr,
                        group_sum.data_ptr<_Float32>(), size);
  return group_sum;
}

torch::Tensor GraphL2NormCUDA(torch::Tensor indptr,
                              torch::optional<torch::Tensor> e_ids,
                              torch::optional<torch::Tensor> data) {
  auto size = indptr.numel() - 1;
  auto data_ptr =
      (data.has_value()) ? data.value().data_ptr<_Float32>() : nullptr;
  auto e_ids_ptr =
      (e_ids.has_value()) ? e_ids.value().data_ptr<int64_t>() : nullptr;
  auto options = (data.has_value())
                     ? data.value().options()
                     : torch::dtype(torch::kFloat32).device(torch::kCUDA);
  auto group_norm = torch::zeros(size, options);

  dim3 block(32, 16);
  dim3 grid((size + block.x - 1) / block.x);
  GroupNormL2<int64_t, _Float32>
      <<<grid, block>>>(indptr.data_ptr<int64_t>(), e_ids_ptr, data_ptr,
                        group_norm.data_ptr<_Float32>(), size);
  return group_norm;
}

torch::Tensor GraphDivCUDA(torch::Tensor indptr,
                           torch::optional<torch::Tensor> e_ids,
                           torch::optional<torch::Tensor> data,
                           torch::Tensor divisor) {
  auto size = indptr.numel() - 1;
  auto data_ptr =
      (data.has_value()) ? data.value().data_ptr<_Float32>() : nullptr;
  auto e_ids_ptr =
      (e_ids.has_value()) ? e_ids.value().data_ptr<int64_t>() : nullptr;
  auto options = (data.has_value())
                     ? data.value().options()
                     : torch::dtype(torch::kFloat32).device(torch::kCUDA);
  thrust::device_ptr<int64_t> item_prefix(
      static_cast<int64_t*>(indptr.data_ptr<int64_t>()));
  int64_t n_edges = item_prefix[size];
  auto out_data = torch::zeros(n_edges, options);

  dim3 block(32, 16);
  dim3 grid((size + block.x - 1) / block.x);
  GroupDiv<int64_t, _Float32><<<grid, block>>>(
      indptr.data_ptr<int64_t>(), e_ids_ptr, data_ptr,
      divisor.data_ptr<_Float32>(), out_data.data_ptr<_Float32>(), size);
  return out_data;
}

torch::Tensor GraphNormalizeCUDA(torch::Tensor indptr,
                                 torch::optional<torch::Tensor> e_ids,
                                 torch::optional<torch::Tensor> data) {
  torch::Tensor out_data, group_sum;
  group_sum = GraphSumCUDA(indptr, e_ids, data);
  out_data = GraphDivCUDA(indptr, e_ids, data, group_sum);
  return out_data;
}
}  // namespace impl
}  // namespace gs