#include "graph_ops.h"

#include "atomic.h"
#include "cuda_common.h"
#include "utils.h"

namespace gs {
namespace impl {
template <typename IdType, typename DType>
__global__ void GroupSum(IdType* indptr, DType* data, DType* out_data,
                         int64_t size) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;

  while (row < size) {
    IdType start = indptr[row];
    IdType n_edges = indptr[row + 1] - start;
    for (int idx = threadIdx.x; idx < n_edges; idx += blockDim.x) {
      AtomicAdd(&out_data[row], data[start + idx]);
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType, typename DType>
__global__ void GroupDiv(IdType* indptr, DType* data, DType* divisor,
                         DType* out_data, int64_t size) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;

  while (row < size) {
    IdType start = indptr[row];
    IdType n_edges = indptr[row + 1] - start;
    for (int idx = threadIdx.x; idx < n_edges; idx += blockDim.x) {
      out_data[start + idx] = data[start + idx] / divisor[row];
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType, typename DType>
__global__ void GroupDiv_2index(IdType* indptr, IdType* e_ids, DType* data,
                                DType* divisor, DType* out_data, int64_t size) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;

  while (row < size) {
    IdType start = indptr[row];
    IdType n_edges = indptr[row + 1] - start;
    IdType pos;
    for (int idx = threadIdx.x; idx < n_edges; idx += blockDim.x) {
      pos = e_ids[start + idx];
      out_data[pos] = data[pos] / divisor[row];
    }
    row += gridDim.x * blockDim.y;
  }
}

template <typename IdType, typename DType>
__global__ void GroupNormL2(IdType* indptr, DType* data, DType* out_data,
                            int64_t size) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;

  while (row < size) {
    IdType start = indptr[row];
    IdType n_edges = indptr[row + 1] - start;
    for (int idx = threadIdx.x; idx < n_edges; idx += blockDim.x) {
      AtomicAdd(&out_data[row], powf(data[start + idx], 2));
    }
    row += gridDim.x * blockDim.y;
  }
}

torch::Tensor GraphSumCUDA(torch::Tensor indptr, torch::Tensor data,
                           torch::optional<torch::Tensor> e_ids) {
  auto size = indptr.numel() - 1;
  auto group_sum = torch::zeros(size, data.options());
  dim3 block(32, 16);
  dim3 grid((size + block.x - 1) / block.x);
  GroupSum<int64_t, _Float32>
      <<<grid, block>>>(indptr.data_ptr<int64_t>(), data.data_ptr<_Float32>(),
                        group_sum.data_ptr<_Float32>(), size);
  return group_sum;
}

torch::Tensor GraphL2NormCUDA(torch::Tensor indptr, torch::Tensor data) {
  auto size = indptr.numel() - 1;
  auto group_norm = torch::zeros(size, data.options());
  dim3 block(32, 16);
  dim3 grid((size + block.x - 1) / block.x);
  GroupNormL2<int64_t, _Float32>
      <<<grid, block>>>(indptr.data_ptr<int64_t>(), data.data_ptr<_Float32>(),
                        group_norm.data_ptr<_Float32>(), size);
  return group_norm;
}

torch::Tensor GraphDivCUDA(torch::Tensor indptr, torch::Tensor data,
                           torch::Tensor divisor) {
  auto size = indptr.numel() - 1;
  auto out_data = torch::zeros(data.numel(), data.options());
  dim3 block(32, 16);
  dim3 grid((size + block.x - 1) / block.x);
  GroupDiv<int64_t, _Float32><<<grid, block>>>(
      indptr.data_ptr<int64_t>(), data.data_ptr<_Float32>(),
      divisor.data_ptr<_Float32>(), out_data.data_ptr<_Float32>(), size);
  return out_data;
}

torch::Tensor GraphDivCUDA_2index(torch::Tensor indptr, torch::Tensor data,
                                  torch::Tensor e_ids, torch::Tensor divisor) {
  auto size = indptr.numel() - 1;
  auto out_data = torch::zeros(data.numel(), data.options());
  dim3 block(32, 16);
  dim3 grid((size + block.x - 1) / block.x);
  GroupDiv_2index<int64_t, _Float32>
      <<<grid, block>>>(indptr.data_ptr<int64_t>(), e_ids.data_ptr<int64_t>(),
                        data.data_ptr<_Float32>(), divisor.data_ptr<_Float32>(),
                        out_data.data_ptr<_Float32>(), size);
  return out_data;
}

torch::Tensor GraphNormalizeCUDA(torch::Tensor indptr, torch::Tensor data) {
  torch::Tensor out_data, group_sum;
  group_sum = GraphSumCUDA(indptr, data);
  out_data = GraphDivCUDA(indptr, data, group_sum);
  return out_data;
}
}  // namespace impl
}  // namespace gs