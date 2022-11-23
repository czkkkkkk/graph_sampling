#ifndef GS_CUDA_MACRO_H_
#define GS_CUDA_MACRO_H_

#include "./cuda_common.h"

/* Macro used for switching between broadcasting and non-broadcasting kernels.
 * It also copies the auxiliary information for calculating broadcasting offsets
 * to GPU.
 */
#define BCAST_IDX_CTX_SWITCH(BCAST, EDGE_MAP, LHS_OFF, RHS_OFF, ...)   \
  do {                                                                 \
    const BcastOff &info = (BCAST);                                    \
    if (!info.use_bcast) {                                             \
      constexpr bool UseBcast = false;                                 \
      if ((EDGE_MAP)) {                                                \
        constexpr bool UseIdx = true;                                  \
        { __VA_ARGS__ }                                                \
      } else {                                                         \
        constexpr bool UseIdx = false;                                 \
        { __VA_ARGS__ }                                                \
      }                                                                \
    } else {                                                           \
      constexpr bool UseBcast = true;                                  \
      CUDA_CALL(cudaMalloc((void **)&LHS_OFF,                          \
                           sizeof(int64_t) * info.lhs_offset.size())); \
      CUDA_CALL(cudaMemcpy((LHS_OFF), &info.lhs_offset[0],             \
                           sizeof(int64_t) * info.lhs_offset.size(),   \
                           cudaMemcpyHostToDevice));                   \
      CUDA_CALL(cudaMalloc((void **)&RHS_OFF,                          \
                           sizeof(int64_t) * info.rhs_offset.size())); \
      CUDA_CALL(cudaMemcpy((RHS_OFF), &info.rhs_offset[0],             \
                           sizeof(int64_t) * info.rhs_offset.size(),   \
                           cudaMemcpyHostToDevice));                   \
      if ((EDGE_MAP)) {                                                \
        constexpr bool UseIdx = true;                                  \
        { __VA_ARGS__ }                                                \
      } else {                                                         \
        constexpr bool UseIdx = false;                                 \
        { __VA_ARGS__ }                                                \
      }                                                                \
      CUDA_CALL(cudaFree(LHS_OFF));                                    \
      CUDA_CALL(cudaFree(RHS_OFF));                                    \
    }                                                                  \
  } while (0)

#endif
