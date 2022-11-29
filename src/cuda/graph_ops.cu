#include "graph_ops.h"

#include <curand_kernel.h>
#include <nvToolsExt.h>
#include "atomic.h"
#include "cuda_common.h"
#include "utils.h"

namespace gs {
namespace impl {
template <typename IdType, typename DType>
__global__ void _SegmentDivKernel(IdType* indptr, IdType* e_ids, DType* data,
                                  DType* divisor, DType* out_data,
                                  int64_t size) {
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

/**
 * @brief SpMV for graphSum
 */
template <typename IdType, typename DType, int BLOCK_WARPS, int TILE_SIZE>
__global__ void _SegmentSumKernel(IdType* indptr, DType* data, int size,
                                  int powk, DType* out) {
  assert(blockDim.x == WARP_SIZE);
  assert(blockDim.y == BLOCK_WARPS);
  assert(powk > 0);

  int warp_id = threadIdx.y;
  int laneid = threadIdx.x;
  IdType out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  IdType last_row =
      MIN(static_cast<IdType>((blockIdx.x + 1) * TILE_SIZE), size);

  typedef cub::WarpReduce<DType> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage[BLOCK_WARPS];

  while (out_row < last_row) {
    DType local_reduce = 0;
    IdType in_row_start = indptr[out_row];
    IdType in_row_end = indptr[out_row + 1];
    for (int idx = in_row_start + laneid; idx < in_row_end; idx += WARP_SIZE) {
      local_reduce += powf(data[idx], powk);
    }

    DType reduce = WarpReduce(temp_storage[warp_id]).Sum(local_reduce);
    if (laneid == 0) {
      out[out_row] = reduce;
    }

    out_row += BLOCK_WARPS;
  }
}

template <typename IdType, typename DType>
torch::Tensor GraphSum(torch::Tensor indptr,
                       torch::optional<torch::Tensor> e_ids,
                       torch::optional<torch::Tensor> data, int64_t powk) {
  auto size = indptr.numel() - 1;
  auto options = (data.has_value())
                     ? data.value().options()
                     : torch::dtype(torch::kFloat32).device(torch::kCUDA);
  torch::Tensor segment_sum = torch::empty(size, options);

  if (data.has_value()) {
    auto permuted_data = (e_ids.has_value())
                             ? data.value().index({e_ids.value()})
                             : data.value();

    // Aligning DGL
    constexpr int TILE_SIZE = 256;
    constexpr int BLOCK_WARPS = 256 / WARP_SIZE;
    int nb = (size + TILE_SIZE - 1) / TILE_SIZE;
    const dim3 block(WARP_SIZE, BLOCK_WARPS);
    const dim3 grid(nb);
    _SegmentSumKernel<IdType, DType, BLOCK_WARPS, TILE_SIZE><<<grid, block>>>(
        indptr.data_ptr<IdType>(), permuted_data.data_ptr<DType>(), size, powk,
        segment_sum.data_ptr<DType>());

  } else {
    using it = thrust::counting_iterator<IdType>;
    thrust::for_each(
        thrust::device, it(0), it(size),
        [d_offsets = indptr.data_ptr<IdType>(),
         out = segment_sum.data_ptr<DType>()] __device__(int i) mutable {
          out[i] = static_cast<DType>(d_offsets[i + 1] - d_offsets[i]);
        });
  }
  return segment_sum;
}

torch::Tensor GraphSumCUDA(torch::Tensor indptr,
                           torch::optional<torch::Tensor> e_ids,
                           torch::optional<torch::Tensor> data, int64_t powk) {
  return GraphSum<int64_t, float>(indptr, e_ids, data, powk);
}

template <typename IdType, typename DType>
torch::Tensor GraphDiv(torch::Tensor indptr,
                       torch::optional<torch::Tensor> e_ids,
                       torch::optional<torch::Tensor> data,
                       torch::Tensor divisor) {
  auto size = indptr.numel() - 1;
  auto data_ptr = (data.has_value()) ? data.value().data_ptr<DType>() : nullptr;
  auto e_ids_ptr =
      (e_ids.has_value()) ? e_ids.value().data_ptr<IdType>() : nullptr;
  auto options = (data.has_value())
                     ? data.value().options()
                     : torch::dtype(torch::kFloat32).device(torch::kCUDA);
  thrust::device_ptr<IdType> item_prefix(
      static_cast<IdType*>(indptr.data_ptr<IdType>()));
  int64_t n_edges = item_prefix[size];
  auto out_data = torch::zeros(n_edges, options);

  dim3 block(32, 16);
  dim3 grid((size + block.x - 1) / block.x);
  _SegmentDivKernel<IdType, DType><<<grid, block>>>(
      indptr.data_ptr<IdType>(), e_ids_ptr, data_ptr, divisor.data_ptr<DType>(),
      out_data.data_ptr<DType>(), size);
  return out_data;
}

torch::Tensor GraphDivCUDA(torch::Tensor indptr,
                           torch::optional<torch::Tensor> e_ids,
                           torch::optional<torch::Tensor> data,
                           torch::Tensor divisor) {
  return GraphDiv<int64_t, float>(indptr, e_ids, data, divisor);
}

template <typename IdType, typename DType>
torch::Tensor GraphNormalize(torch::Tensor indptr,
                             torch::optional<torch::Tensor> e_ids,
                             torch::optional<torch::Tensor> data) {
  torch::Tensor out_data, segment_sum;
  segment_sum = GraphSum<IdType, DType>(indptr, e_ids, data, 1);
  out_data = GraphDiv<IdType, DType>(indptr, e_ids, data, segment_sum);
  return out_data;
}

torch::Tensor GraphNormalizeCUDA(torch::Tensor indptr,
                                 torch::optional<torch::Tensor> e_ids,
                                 torch::optional<torch::Tensor> data) {
  return GraphNormalize<int64_t, float>(indptr, e_ids, data);
}
}  // namespace impl
}  // namespace gs