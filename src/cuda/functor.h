#ifndef GS_CUDA_FUNCTOR_H_
#define GS_CUDA_FUNCTOR_H_

#include <cuda_runtime.h>

namespace gs {
namespace impl {
//////////// CUDA binary operators ////////////////
namespace binary {
template <typename DType>
struct Add {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  static constexpr bool reduce_last_dim = false;
  static __device__ __forceinline__ DType Call(const DType *lhs,
                                               const DType *rhs,
                                               int64_t len = 1) {
    return lhs[0] + rhs[0];
  }
};
template <typename DType>
constexpr bool Add<DType>::use_lhs;
template <typename DType>
constexpr bool Add<DType>::use_rhs;
template <typename DType>
constexpr bool Add<DType>::reduce_last_dim;

template <typename DType>
struct Sub {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  static constexpr bool reduce_last_dim = false;
  static __device__ __forceinline__ DType Call(const DType *lhs,
                                               const DType *rhs,
                                               int64_t len = 1) {
    return lhs[0] - rhs[0];
  }
};
template <typename DType>
constexpr bool Sub<DType>::use_lhs;
template <typename DType>
constexpr bool Sub<DType>::use_rhs;
template <typename DType>
constexpr bool Sub<DType>::reduce_last_dim;

template <typename DType>
struct Mul {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  static constexpr bool reduce_last_dim = false;
  static __device__ __forceinline__ DType Call(const DType *lhs,
                                               const DType *rhs,
                                               int64_t len = 1) {
    return lhs[0] * rhs[0];
  }
};
template <typename DType>
constexpr bool Mul<DType>::use_lhs;
template <typename DType>
constexpr bool Mul<DType>::use_rhs;
template <typename DType>
constexpr bool Mul<DType>::reduce_last_dim;

template <typename DType>
struct Div {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  static constexpr bool reduce_last_dim = false;
  static __device__ __forceinline__ DType Call(const DType *lhs,
                                               const DType *rhs,
                                               int64_t len = 1) {
    return lhs[0] / rhs[0];
  }
};
template <typename DType>
constexpr bool Div<DType>::use_lhs;
template <typename DType>
constexpr bool Div<DType>::use_rhs;
template <typename DType>
constexpr bool Div<DType>::reduce_last_dim;

template <typename DType>
struct CopyLhs {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = false;
  static constexpr bool reduce_last_dim = false;
  static __device__ __forceinline__ DType Call(const DType *lhs,
                                               const DType *rhs,
                                               int64_t len = 1) {
    return lhs[0];
  }
};
template <typename DType>
constexpr bool CopyLhs<DType>::use_lhs;
template <typename DType>
constexpr bool CopyLhs<DType>::use_rhs;
template <typename DType>
constexpr bool CopyLhs<DType>::reduce_last_dim;

template <typename DType>
struct CopyRhs {
  static constexpr bool use_lhs = false;
  static constexpr bool use_rhs = true;
  static constexpr bool reduce_last_dim = false;
  static __device__ __forceinline__ DType Call(const DType *lhs,
                                               const DType *rhs,
                                               int64_t len = 1) {
    return rhs[0];
  }
};
template <typename DType>
constexpr bool CopyRhs<DType>::use_lhs;
template <typename DType>
constexpr bool CopyRhs<DType>::use_rhs;
template <typename DType>
constexpr bool CopyRhs<DType>::reduce_last_dim;

template <typename DType>
struct Dot {
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  static constexpr bool reduce_last_dim = true;
  static __device__ __forceinline__ DType Call(const DType *lhs,
                                               const DType *rhs,
                                               int64_t len = 1) {
    DType rst = static_cast<DType>(0);
    for (int64_t i = 0; i < len; ++i) {
      rst += lhs[i] * rhs[i];
    }
    return rst;
  }
};
template <typename DType>
constexpr bool Dot<DType>::use_lhs;
template <typename DType>
constexpr bool Dot<DType>::use_rhs;
template <typename DType>
constexpr bool Dot<DType>::reduce_last_dim;

}  // end of namespace binary

}  // namespace impl
}  // namespace gs

#endif  // GS_CUDA_FUNCTOR_CUH_
