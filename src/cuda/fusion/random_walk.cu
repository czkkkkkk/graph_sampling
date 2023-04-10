#include <curand_kernel.h>
#include "../cuda_common.h"
#include "../utils.h"
#include "random_walk.h"
#include <nvToolsExt.h>

namespace gs {
namespace impl {
namespace fusion {

  template <int BLOCK_SIZE>
  __global__ void _RandomWalkKernel_prefix(const int32_t* seed_data,
                                    const int32_t num_seeds,
                                    const uint32_t max_num_steps,
                                    int32_t* graph_indice, int32_t* graph_indptr,
                                    int32_t* out_traces_data) {

  }

template <int BLOCK_SIZE>
__global__ void _RandomWalkKernel(const int32_t* seed_data,
                                  const int32_t num_seeds,
                                  const uint32_t max_num_steps,
                                  int32_t* graph_indice, int32_t* graph_indptr,
                                  int32_t* out_traces_data) {
  int32_t tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int32_t last_idx =
      min(static_cast<int32_t>(blockIdx.x + 1) * BLOCK_SIZE, num_seeds);     
  for (int idx = tid; idx < last_idx; idx += BLOCK_SIZE) {
    int32_t curr = seed_data[idx];
    out_traces_data[0 * num_seeds + idx] = curr;
  }
  curandState rng;
  uint32_t rand_seed = 7777777;
  curand_init(rand_seed + tid, 0, 0, &rng);
  int32_t grid_mul_block = gridDim.x * blockDim.x; 
  // begin random walk
  for (int idx = tid; idx < num_seeds; idx += grid_mul_block) {
  for (int step_idx = 0; step_idx < max_num_steps; step_idx++) {


      int32_t curr = out_traces_data[step_idx * num_seeds + idx];

      int32_t pick = -1;
      if (curr == -1) {
        out_traces_data[(step_idx + 1) * num_seeds + idx] = pick;
      } else {
        const int32_t in_row_start = graph_indptr[curr];
        const int32_t deg = graph_indptr[curr + 1] - graph_indptr[curr];
        if (deg > 0) {
          pick = graph_indice[in_row_start + curand(&rng) % deg];
          out_traces_data[(step_idx + 1) * num_seeds + idx] = pick;
        }
        else{
          out_traces_data[(step_idx + 1) * num_seeds + idx] = -1;
        }
       }
    }
  }
}

// torch::Tensor FusedRandomWalkCUDA(torch::Tensor seeds, int64_t walk_length,
//                                   int64_t* indices, int64_t* indptr) {
//   const int64_t* seed_data = seeds.data_ptr<int64_t>();
//   const int64_t num_seeds = seeds.numel();
//   const uint64_t max_num_steps = walk_length;
//   int64_t outsize = num_seeds * (max_num_steps + 1);
//   torch::Tensor out_traces_tensor = torch::empty(outsize, seeds.options());

//   int64_t* out_traces_data = out_traces_tensor.data_ptr<int64_t>();
//   constexpr int BLOCK_SIZE = 256;
//   dim3 block(BLOCK_SIZE);
//   // dim3 grid((num_seeds + BLOCK_SIZE - 1) / BLOCK_SIZE);
//   dim3 grid(320);

//   nvtxRangePushA("RandomWalkKernel_prefix");   
//   _RandomWalkKernel_prefix<BLOCK_SIZE>
//       <<<grid, block>>>(seeds.data_ptr<int64_t>(), num_seeds, max_num_steps,
//                         indices, indptr, out_traces_data);
//   nvtxRangePop();
//   nvtxRangePushA("RandomWalkKernel_randomwalk");                                  
//   _RandomWalkKernel<BLOCK_SIZE>
//       <<<grid, block>>>(seeds.data_ptr<int64_t>(), num_seeds, max_num_steps,
//                         indices, indptr, out_traces_data);
//   nvtxRangePop();
//   return out_traces_tensor.reshape({seeds.numel(), -1});
// }

torch::Tensor FusedRandomWalkCUDA(torch::Tensor seeds, int64_t walk_length,
  int32_t* indices, int32_t* indptr) {
const int32_t* seed_data = seeds.data_ptr<int32_t>();
const int32_t num_seeds = seeds.numel();
const uint32_t max_num_steps = (uint32_t)walk_length;
int32_t outsize = num_seeds * (max_num_steps + 1);
torch::Tensor out_traces_tensor = torch::empty(outsize, seeds.options());

int32_t* out_traces_data = out_traces_tensor.data_ptr<int32_t>();
constexpr int BLOCK_SIZE = 256;
dim3 block(BLOCK_SIZE);
// dim3 grid((num_seeds + BLOCK_SIZE - 1) / BLOCK_SIZE);
dim3 grid(320);
 
// _RandomWalkKernel_prefix<BLOCK_SIZE>
// <<<grid, block>>>(seeds.data_ptr<int32_t>(), num_seeds, max_num_steps,
// indices, indptr, out_traces_data);                             
_RandomWalkKernel<BLOCK_SIZE>
<<<grid, block>>>(seeds.data_ptr<int32_t>(), num_seeds, max_num_steps,
indices, indptr, out_traces_data);
return out_traces_tensor.reshape({seeds.numel(), -1});
}

}  // namespace fusion
}  // namespace impl
}  // namespace gs