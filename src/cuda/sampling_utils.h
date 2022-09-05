#ifndef GS_CUDA_SAMPLING_UTILS_H_
#define GS_CUDA_SAMPLING_UTILS_H_

#include "cuda_common.h"
#include "utils.h"

namespace gs {
namespace impl {

template <typename IdType>
torch::Tensor GetSampledSubIndptr(torch::Tensor indptr, int64_t fanout,
                                  bool replace) {
  int64_t size = indptr.numel();
  auto new_indptr = torch::zeros(size, indptr.options());
  thrust::device_ptr<IdType> item_prefix(
      static_cast<IdType*>(new_indptr.data_ptr<IdType>()));

  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(
      thrust::device, it(0), it(size),
      [in_indptr = indptr.data_ptr<IdType>(),
       out = thrust::raw_pointer_cast(item_prefix), if_replace = replace,
       num_fanout = fanout] __device__(int i) mutable {
        IdType begin = in_indptr[i];
        IdType end = in_indptr[i + 1];
        if (if_replace) {
          out[i] = (end - begin) == 0 ? 0 : num_fanout;
        } else {
          out[i] = min(end - begin, num_fanout);
        }
      });

  cub_exclusiveSum<IdType>(thrust::raw_pointer_cast(item_prefix), size + 1);
  return new_indptr;
}

template <typename IdType>
torch::Tensor GetSampledSubIndptrFused(torch::Tensor indptr,
                                       torch::Tensor column_ids, int64_t fanout,
                                       bool replace) {
  int64_t size = column_ids.numel();
  auto sub_indptr = torch::empty(size + 1, indptr.options());
  thrust::device_ptr<IdType> item_prefix(
      static_cast<IdType*>(sub_indptr.data_ptr<IdType>()));

  using it = thrust::counting_iterator<IdType>;
  thrust::for_each(
      thrust::device, it(0), it(size),
      [in = column_ids.data_ptr<IdType>(),
       in_indptr = indptr.data_ptr<IdType>(),
       out = thrust::raw_pointer_cast(item_prefix), if_replace = replace,
       num_fanout = fanout] __device__(int i) mutable {
        IdType begin = in_indptr[in[i]];
        IdType end = in_indptr[in[i] + 1];
        if (if_replace) {
          out[i] = (end - begin) == 0 ? 0 : num_fanout;
        } else {
          out[i] = min(end - begin, num_fanout);
        }
      });

  cub_exclusiveSum<IdType>(thrust::raw_pointer_cast(item_prefix), size + 1);
  return sub_indptr;
}
}  // namespace impl
}  // namespace gs

#endif
