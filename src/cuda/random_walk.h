#ifndef GS_CUDA_RANDOM_WALK_H_
#define GS_CUDA_RANDOM_WALK_H_
#include <torch/torch.h>

namespace gs {
namespace impl {
torch::Tensor RandomWalkFusedCUDA(torch::Tensor seeds, int64_t walk_length,
                                  int64_t* indices, int64_t* indptr);
}
}

#endif