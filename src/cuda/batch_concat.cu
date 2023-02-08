#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "atomic.h"
#include "cuda_common.h"
#include "tensor_ops.h"
#include "utils.h"

namespace gs {
namespace impl {

template <typename IdType>
__global__ void _BatchConcatKernel(IdType** __restrict__ data_tensors,
                                   IdType** __restrict__ data_ptrs,
                                   IdType* __restrict__ out,
                                   IdType* __restrict__ key,
                                   IdType* __restrict__ out_ptr,
                                   int64_t num_segments, int64_t num_batchs) {
  assert(num_batchs < 8);
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t acc_offset[8] = {0, 0, 0, 0, 0, 0, 0, 0};

  for (int64_t i = 0; i < num_segments; i++) {
    int64_t data_len = data_ptrs[i][num_batchs];

    for (int64_t index = tid; index < data_len;
         index += gridDim.x * blockDim.x) {
      int64_t batch_index = cub::UpperBound<IdType*, int64_t, IdType>(
                                data_ptrs[i], num_batchs + 1, index) -
                            1;
      int64_t out_begin = out_ptr[batch_index];
      int64_t out_index = index - data_ptrs[i][batch_index];
      out[out_begin + acc_offset[batch_index] + out_index] =
          data_tensors[i][index];
      key[out_begin + acc_offset[batch_index] + out_index] = batch_index;
    }

    for (int64_t k = 0; k < num_batchs; k++) {
      acc_offset[k] += data_ptrs[i][k + 1] - data_ptrs[i][k];
    }
  }
}

template <typename IdType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> _BatchConcat(
    const std::vector<torch::Tensor>& data_tensors,
    const std::vector<torch::Tensor>& offset_tensors) {
  int64_t num_segments = data_tensors.size();
  int64_t num_batchs = offset_tensors[0].numel() - 1;
  int64_t max_tensor_len = 0;

  thrust::host_vector<IdType*> h_data_tensor_ptrs(num_segments);
  thrust::host_vector<IdType*> h_offset_tensor_ptrs(num_segments);

  for (int i = 0; i < num_segments; i++) {
    h_data_tensor_ptrs[i] = data_tensors[i].data_ptr<IdType>();
    h_offset_tensor_ptrs[i] = offset_tensors[i].data_ptr<IdType>();
    max_tensor_len = max(max_tensor_len, data_tensors[i].numel());
  }

  thrust::device_vector<IdType*> d_data_tensor_ptrs = h_data_tensor_ptrs;
  thrust::device_vector<IdType*> d_offset_tensor_ptrs = h_offset_tensor_ptrs;

  torch::Tensor total_tensor_ptr = torch::empty(
      num_batchs + 1,
      torch::dtype(offset_tensors[0].dtype()).device(torch::kCUDA));

  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(
      it(0), it(num_batchs + 1),
      [in = d_offset_tensor_ptrs.data(), num_segments,
       out = total_tensor_ptr.data_ptr<IdType>()] __device__(IdType i) mutable {
        IdType sum = 0;
        for (int j = 0; j < num_segments; j++) {
          sum += in[j][i];
        }
        out[i] = sum;
      });

  thrust::device_ptr<IdType> wrapper_total_tensor_ptr(
      static_cast<IdType*>(total_tensor_ptr.data_ptr<IdType>()));
  int64_t total_size = wrapper_total_tensor_ptr[num_batchs];
  torch::Tensor total_tensor =
      torch::empty(total_size, data_tensors[0].options());
  torch::Tensor key_tensor =
      torch::empty(total_size, data_tensors[0].options());

  constexpr int BLOCK_SIZE = 256;
  int num_blocks = (max_tensor_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 blocks(BLOCK_SIZE);
  dim3 grid(num_blocks);
  _BatchConcatKernel<IdType><<<grid, blocks>>>(
      thrust::raw_pointer_cast(d_data_tensor_ptrs.data()),
      thrust::raw_pointer_cast(d_offset_tensor_ptrs.data()),
      total_tensor.data_ptr<IdType>(), key_tensor.data_ptr<IdType>(),
      total_tensor_ptr.data_ptr<IdType>(), num_segments, num_batchs);

  return {total_tensor, key_tensor, total_tensor_ptr};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> BatchConcatCUDA(
    const std::vector<torch::Tensor>& data_tensors,
    const std::vector<torch::Tensor>& offset_tensors) {
  return _BatchConcat<int64_t>(data_tensors, offset_tensors);
}

}  // namespace impl
}  // namespace gs
