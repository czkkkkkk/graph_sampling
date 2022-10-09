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

template <typename IdType, typename DType>
torch::Tensor GraphSum(torch::Tensor indptr,
                       torch::optional<torch::Tensor> e_ids,
                       torch::optional<torch::Tensor> data, int64_t powk) {
  auto size = indptr.numel() - 1;
  auto options = (data.has_value())
                     ? data.value().options()
                     : torch::dtype(torch::kFloat32).device(torch::kCUDA);
  auto segment_sum = torch::zeros(size, options);

  if (data.has_value()) {
    auto permuted_data = (e_ids.has_value())
                             ? data.value().index({e_ids.value()})
                             : data.value();
    auto data_powk = permuted_data;
    if (powk != 1) {
      using it = thrust::counting_iterator<IdType>;
      thrust::for_each(
          thrust::device, it(0), it(permuted_data.numel()),
          [in = permuted_data.data_ptr<DType>(),
           out = data_powk.data_ptr<DType>(),
           k = powk] __device__(int i) mutable { out[i] = powf(in[i], k); });
    }
    cub_segmentedSum<IdType, DType>(data_powk.data_ptr<DType>(),
                                    segment_sum.data_ptr<DType>(),
                                    indptr.data_ptr<IdType>(), size);
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

template <int BLOCK_SIZE>
__global__ void _RandomWalkKernel(const int64_t* seed_data,
                                  const int64_t num_seeds,
                                  const uint64_t max_num_steps,
                                  int64_t* graph_indice, int64_t* graph_indptr,
                                  int64_t* out_traces_data) {
  int64_t tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int64_t last_idx =
      min(static_cast<int64_t>(blockIdx.x + 1) * BLOCK_SIZE, num_seeds);
  curandState rng;
  uint64_t rand_seed = 7777777;
  curand_init(rand_seed + tid, 0, 0, &rng);

  // fisrt step
  for (int idx = tid; idx < last_idx; idx += BLOCK_SIZE) {
    int64_t curr = seed_data[idx];
    out_traces_data[0 * num_seeds + idx] = curr;
  }

  // begin random walk
  for (int step_idx = 0; step_idx < max_num_steps; step_idx++) {
    for (int idx = tid; idx < last_idx; idx += BLOCK_SIZE) {
      int64_t curr = out_traces_data[step_idx * num_seeds + idx];

      int64_t pick = -1;
      if (curr < 0) {
        out_traces_data[(step_idx + 1) * num_seeds + idx] = pick;
      } else {
        const int64_t in_row_start = graph_indptr[curr];
        const int64_t deg = graph_indptr[curr + 1] - graph_indptr[curr];

        if (deg > 0) {
          pick = graph_indice[in_row_start + curand(&rng) % deg];
        }
        out_traces_data[(step_idx + 1) * num_seeds + idx] = pick;
      }
    }
  }
}

torch::Tensor RandomWalkFusedCUDA(torch::Tensor seeds, int64_t walk_length,
                                  int64_t* indices, int64_t* indptr) {
  const int64_t* seed_data = seeds.data_ptr<int64_t>();
  const int64_t num_seeds = seeds.numel();
  const uint64_t max_num_steps = walk_length;
  int64_t outsize = num_seeds * (max_num_steps + 1);
  torch::Tensor out_traces_tensor = torch::empty(outsize, seeds.options());

  int64_t* out_traces_data = out_traces_tensor.data_ptr<int64_t>();
  constexpr int BLOCK_SIZE = 256;
  dim3 block(BLOCK_SIZE);
  dim3 grid((num_seeds + BLOCK_SIZE - 1) / BLOCK_SIZE);
  _RandomWalkKernel<BLOCK_SIZE>
      <<<grid, block>>>(seeds.data_ptr<int64_t>(), num_seeds, max_num_steps,
                        indices, indptr, out_traces_data);
  return out_traces_tensor.reshape({seeds.numel(), -1});
}

torch::Tensor GraphNormalizeCUDA(torch::Tensor indptr,
                                 torch::optional<torch::Tensor> e_ids,
                                 torch::optional<torch::Tensor> data) {
  return GraphNormalize<int64_t, float>(indptr, e_ids, data);
}
}  // namespace impl
}  // namespace gs