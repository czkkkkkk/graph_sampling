#ifndef GS_CUDA_SDDMM_H_
#define GS_CUDA_SDDMM_H_

#include "../bcast.h"
#include "../graph_storage.h"
#include "./functor.h"
#include "./selector.h"

namespace gs {
namespace impl {

#define SWITCH_OP(op, Op, ...)                                        \
  do {                                                                \
    if ((op) == "add") {                                              \
      typedef impl::binary::Add<DType> Op;                            \
      { __VA_ARGS__ }                                                 \
    } else if ((op) == "sub") {                                       \
      typedef impl::binary::Sub<DType> Op;                            \
      { __VA_ARGS__ }                                                 \
    } else if ((op) == "mul") {                                       \
      typedef impl::binary::Mul<DType> Op;                            \
      { __VA_ARGS__ }                                                 \
    } else if ((op) == "div") {                                       \
      typedef impl::binary::Div<DType> Op;                            \
      { __VA_ARGS__ }                                                 \
    } else if ((op) == "copy_lhs") {                                  \
      typedef impl::binary::CopyLhs<DType> Op;                        \
      { __VA_ARGS__ }                                                 \
    } else if ((op) == "copy_rhs") {                                  \
      typedef impl::binary::CopyRhs<DType> Op;                        \
      { __VA_ARGS__ }                                                 \
    } else if ((op) == "dot") {                                       \
      typedef impl::binary::Dot<DType> Op;                            \
      { __VA_ARGS__ }                                                 \
    } else {                                                          \
      LOG(FATAL) << "Unsupported SpMM/SDDMM binary operator: " << op; \
    }                                                                 \
  } while (0)

#define SWITCH_RHS(rhs_target, RhsTarget, ...)             \
  do {                                                     \
    if ((rhs_target) == 0) {                               \
      constexpr int RhsTarget = 0;                         \
      { __VA_ARGS__ }                                      \
    } else if ((rhs_target) == 1) {                        \
      constexpr int RhsTarget = 1;                         \
      { __VA_ARGS__ }                                      \
    } else if ((rhs_target) == 2) {                        \
      constexpr int RhsTarget = 2;                         \
      { __VA_ARGS__ }                                      \
    } else {                                               \
      LOG(INFO) << "Invalid rhs target: " << (rhs_target); \
    }                                                      \
  } while (0)

#define SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, ...) \
  do {                                                                   \
    if ((lhs_target) == 0) {                                             \
      constexpr int LhsTarget = 0;                                       \
      SWITCH_RHS(rhs_target, RhsTarget, __VA_ARGS__);                    \
    } else if ((lhs_target) == 1) {                                      \
      constexpr int LhsTarget = 1;                                       \
      SWITCH_RHS(rhs_target, RhsTarget, __VA_ARGS__);                    \
    } else if ((lhs_target) == 2) {                                      \
      constexpr int LhsTarget = 2;                                       \
      SWITCH_RHS(rhs_target, RhsTarget, __VA_ARGS__);                    \
    } else {                                                             \
      LOG(INFO) << "Invalid lhs target: " << (lhs_target);               \
    }                                                                    \
  } while (0)

void SDDMMCSR(const std::string& op, const BcastOff& bcast,
              std::shared_ptr<CSR> csr, torch::Tensor lhs, torch::Tensor rhs,
              torch::Tensor out, int lhs_target, int rhs_target,
              int64_t num_rows, int64_t num_cols);

void SDDMMCOO(const std::string& op, const BcastOff& bcast,
              std::shared_ptr<COO> coo, torch::Tensor lhs, torch::Tensor rhs,
              torch::Tensor out, int lhs_target, int rhs_target,
              int64_t num_rows, int64_t num_cols);

}  // namespace impl
}  // namespace gs

#endif