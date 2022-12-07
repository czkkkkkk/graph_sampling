#include <curand_kernel.h>
#include <nvToolsExt.h>
#include "cuda_common.h"
#include "heterograph_ops.h"
#include "utils.h"
namespace gs {
namespace impl {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
CSCColumnwiseSamplingOneKeepDimCUDA(torch::Tensor indptr, torch::Tensor indices,
                                    torch::Tensor column_ids) {
  // get subptr
  int64_t num_items = column_ids.numel();
  auto sub_indptr = torch::ones(num_items + 1, indptr.options());
  thrust::device_ptr<int64_t> item_prefix(
      static_cast<int64_t*>(sub_indptr.data_ptr<int64_t>()));
  cub_exclusiveSum<int64_t>(thrust::raw_pointer_cast(item_prefix),
                            num_items + 1);
  auto select_index = torch::empty(num_items, indices.options());
  // get subindices
  auto sub_indices = torch::empty(num_items, indices.options());
  using it = thrust::counting_iterator<int64_t>;
  thrust::for_each(
      thrust::device, it(0), it(num_items),
      [sub_indices_ptr = sub_indices.data_ptr<int64_t>(),
       indptr_ptr = indptr.data_ptr<int64_t>(),
       indices_ptr = indices.data_ptr<int64_t>(),
       sub_indptr_ptr = sub_indptr.data_ptr<int64_t>(),
       select_index_ptr = select_index.data_ptr<int64_t>(),
       column_ids_ptr =
           column_ids.data_ptr<int64_t>()] __device__(int i) mutable {
        const uint64_t random_seed = 7777777;
        curandState rng;
        curand_init(random_seed + i, 0, 0, &rng);
        int64_t col = column_ids_ptr[i];
        int64_t in_start = indptr_ptr[col];
        int64_t out_start = sub_indptr_ptr[i];
        int64_t degree = indptr_ptr[col + 1] - indptr_ptr[col];
        if (degree == 0) {
          sub_indices_ptr[out_start] = -1;
          select_index_ptr[out_start] = -1;
        } else {
          // Sequential Sampling
          // const int64_t edge = tid % degree;
          // Random Sampling
          const int64_t edge = curand(&rng) % degree;
          sub_indices_ptr[out_start] = indices_ptr[in_start + edge];
          select_index_ptr[out_start] = in_start + edge;
        }
      });
  return {sub_indptr, sub_indices, select_index};
}
}  // namespace impl
}  // namespace gs