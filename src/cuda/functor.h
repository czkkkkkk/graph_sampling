#ifndef GS_CUDA_FUNCTOR_H_
#define GS_CUDA_FUNCTOR_H_

#include "./atomic.h"

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

//////////// CUDA reduce operators ////////////////
namespace reduce {
template <typename Idx, typename DType, bool atomic>
struct _Sum {
  static constexpr __host__ __device__ __forceinline__ DType zero() {
    return 0.;
  };
  static constexpr bool require_arg = false;
  static __device__ __forceinline__ void Call(DType *out_buf, Idx *arg_u_buf,
                                              Idx *arg_e_buf, DType val,
                                              Idx uid, Idx eid) {
    if (!atomic) {
      *out_buf += val;
    } else {
      impl::AtomicAdd(out_buf, val);
    }
  }
  static __device__ __forceinline__ void Call(DType *out_buf, Idx *arg_buf,
                                              DType val, Idx id) {
    if (!atomic) {
      *out_buf += val;
    } else {
      impl::AtomicAdd(out_buf, val);
    }
  }
  static __device__ __forceinline__ void CallArg(Idx fid, Idx *arg_u_buf,
                                                 Idx *arg_e_buf, DType val,
                                                 DType val_ref, Idx uid,
                                                 Idx eid) {}
};

template <typename Idx, typename DType, bool atomic = false>
struct Sum : _Sum<Idx, DType, atomic> {};

template <typename Idx, typename DType, bool atomic>
struct _Max {
  static constexpr __host__ __device__ __forceinline__ DType zero() {
    return -std::numeric_limits<DType>::infinity();
  };
  static constexpr bool require_arg = true;
  static __device__ __forceinline__ void Call(DType *out_buf, Idx *arg_u_buf,
                                              Idx *arg_e_buf, DType val,
                                              Idx uid, Idx eid) {
    if (!atomic) {
      if (*out_buf < val) {
        *out_buf = val;
        *arg_u_buf = uid;
        *arg_e_buf = eid;
      }
    } else {
      impl::AtomicMax(out_buf, val);
    }
  }
  static __device__ __forceinline__ void Call(DType *out_buf, Idx *arg_buf,
                                              DType val, Idx id) {
    if (!atomic) {
      if (*out_buf < val) {
        *out_buf = val;
        *arg_buf = id;
      }
    } else {
      impl::AtomicMax(out_buf, val);
    }
  }
  static __device__ __forceinline__ void CallArg(Idx fid, Idx *arg_u_buf,
                                                 Idx *arg_e_buf, DType val,
                                                 DType val_ref, Idx uid,
                                                 Idx eid) {
    if (atomic) {
      if (val == val_ref) {
        if (arg_u_buf) arg_u_buf[fid] = uid;
        if (arg_e_buf) arg_e_buf[fid] = eid;
      }
    }
  }
};

template <typename Idx, typename DType, bool atomic = false>
struct Max : _Max<Idx, DType, atomic> {};

template <typename Idx, typename DType, bool atomic>
struct _Min {
  static constexpr __host__ __device__ __forceinline__ DType zero() {
    return std::numeric_limits<DType>::infinity();
  };
  static constexpr bool require_arg = true;
  static __device__ __forceinline__ void Call(DType *out_buf, Idx *arg_u_buf,
                                              Idx *arg_e_buf, DType val,
                                              Idx uid, Idx eid) {
    if (!atomic) {
      if (*out_buf > val) {
        *out_buf = val;
        *arg_u_buf = uid;
        *arg_e_buf = eid;
      }
    } else {
      impl::AtomicMin(out_buf, val);
    }
  }
  static __device__ __forceinline__ void Call(DType *out_buf, Idx *arg_buf,
                                              DType val, Idx id) {
    if (!atomic) {
      if (*out_buf > val) {
        *out_buf = val;
        *arg_buf = id;
      }
    } else {
      impl::AtomicMin(out_buf, val);
    }
  }
  static __device__ __forceinline__ void CallArg(Idx fid, Idx *arg_u_buf,
                                                 Idx *arg_e_buf, DType val,
                                                 DType val_ref, Idx uid,
                                                 Idx eid) {
    if (atomic) {
      if (val == val_ref) {
        if (arg_u_buf) arg_u_buf[fid] = uid;
        if (arg_e_buf) arg_e_buf[fid] = eid;
      }
    }
  }
};

template <typename Idx, typename DType, bool atomic = false>
struct Min : _Min<Idx, DType, atomic> {};
}  // namespace reduce

}  // namespace impl
}  // namespace gs

#endif  // GS_CUDA_FUNCTOR_CUH_
