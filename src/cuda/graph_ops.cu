#include "graph_ops.h"

namespace gs {
namespace impl {

__global__ void _GetSubSizeKernel(int64_t* sub_indptr, int64_t* indptr,
                                  int64_t* column_ids, int64_t size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    sub_indptr[tid] = 0;
  }
  while (tid < size) {
    int64_t col = column_ids[tid];
    sub_indptr[tid + 1] = indptr[col + 1] - indptr[col];
    tid += gridDim.x * blockDim.x;
  }
}

torch::Tensor GetSubIndptr(torch::Tensor indptr, torch::Tensor ids) {
  int64_t size = ids.numel();
  auto sub_size = torch::empty(size + 1, indptr.options());

  int64_t n_threads = 512;
  int64_t n_blocks = (size + n_threads - 1) / n_threads;
  _GetSubSizeKernel<<<n_blocks, n_threads>>>(sub_size.data_ptr<int64_t>(),
                                             indptr.data_ptr<int64_t>(),
                                             ids.data_ptr<int64_t>(), size);
  return sub_size.cumsum(0);
}

__global__ void _GetSubIndicesKernel(int64_t* out_indices, int64_t* indptr,
                                     int64_t* indices, int64_t* sub_indptr,
                                     int64_t* column_ids, int64_t size) {
  int64_t row = blockIdx.x * blockDim.y + threadIdx.y;

  while (row < size) {
    int64_t in_start = indptr[column_ids[row]];
    int64_t out_start = sub_indptr[row];
    int64_t n_edges = sub_indptr[row + 1] - sub_indptr[row];
    int64_t tid = threadIdx.x;
    while (tid < n_edges) {
      out_indices[out_start + tid] = indices[in_start + tid];
      tid += blockDim.x;
    }
    row += gridDim.x * blockDim.y;
  }
}

torch::Tensor GetSubIndices(torch::Tensor indptr, torch::Tensor indices,
                            torch::Tensor sub_indptr,
                            torch::Tensor column_ids) {
  int64_t size = column_ids.numel();
  // FIXME
  auto n_edges = sub_indptr.to(torch::kCPU).data_ptr<int64_t>()[size];
  auto sub_indices = torch::empty(n_edges, indptr.options());

  dim3 block(32, 8);
  dim3 grid((size + block.x - 1) / block.x);
  _GetSubIndicesKernel<<<grid, block>>>(
      sub_indices.data_ptr<int64_t>(), indptr.data_ptr<int64_t>(),
      indices.data_ptr<int64_t>(), sub_indptr.data_ptr<int64_t>(),
      column_ids.data_ptr<int64_t>(), size);
  return sub_indices;
}

std::pair<torch::Tensor, torch::Tensor> CSCColumnwiseSlicingCUDA(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor column_ids) {
  auto sub_indptr = GetSubIndptr(indptr, column_ids);
  auto sub_indices = GetSubIndices(indptr, indices, sub_indptr, column_ids);
  return {sub_indptr, sub_indices};
}

}  // namespace impl
}  // namespace gs