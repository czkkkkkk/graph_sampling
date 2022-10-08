#include "graph_ops.h"

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

torch::Tensor GraphNormalizeCUDA(torch::Tensor indptr,
                                 torch::optional<torch::Tensor> e_ids,
                                 torch::optional<torch::Tensor> data) {
  return GraphNormalize<int64_t, float>(indptr, e_ids, data);
}
}  // namespace impl
}  // namespace gs