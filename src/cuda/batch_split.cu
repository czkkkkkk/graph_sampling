#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "atomic.h"
#include "cuda_common.h"
#include "tensor_ops.h"
#include "utils.h"

namespace gs {
namespace impl {

template <typename IdType, int64_t BLOCK_WARPS>
__global__ void _SplitIndptrBySize(IdType* indptr, IdType* output, int64_t size,
                                   int64_t num_batchs) {
  assert(blockDim.x == 32);
  assert(blockDim.y == BLOCK_WARPS);

  int laneid = threadIdx.x;
  int warp_id = blockIdx.x * blockDim.y + threadIdx.y;

  for (int i = warp_id; warp_id < num_batchs; i += gridDim.x * blockDim.y) {
    int64_t indptr_begin = size * i;
    int64_t out_begin = (size + 1) * i;
    int64_t offset = indptr[indptr_begin];
    for (int j = laneid; j < size + 1; j += WARP_SIZE) {
      output[j + out_begin] = indptr[j + indptr_begin] - offset;
    }
  }
}

template <typename IdType>
std::vector<torch::Tensor> SplitIndptrBySize(torch::Tensor indptr,
                                             int64_t size) {
  int64_t num_batchs = (indptr.numel() - 1) / size;

  torch::Tensor split_indptr =
      torch::empty(num_batchs * (size + 1), indptr.options());

  constexpr int64_t BLOCK_WARPS = 4;
  dim3 block(WARP_SIZE, BLOCK_WARPS);
  dim3 grid((num_batchs + BLOCK_WARPS - 1) / BLOCK_WARPS);
  /*
  _SplitIndptrBySize<IdType, BLOCK_WARPS>
      <<<grid, block>>>(indptr.data_ptr<IdType>(),
                        split_indptr.data_ptr<IdType>(), size, num_batchs);
    */
  return torch::split(split_indptr, (size + 1));
}

std::vector<torch::Tensor> SplitIndptrBySizeCUDA(torch::Tensor indptr,
                                                 int64_t size) {
  return SplitIndptrBySize<int64_t>(indptr, size);
}
}  // namespace impl
}  // namespace gs