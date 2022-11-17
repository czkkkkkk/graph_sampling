#include "./sddmm.cuh"

namespace gs {
namespace impl {

/**
 * @brief CUDA implementation of g-SDDMM on Csr format.
 */
template <typename IdType, typename DType>
void SDDMMCsr(
    const std::string& op, const BcastOff& bcast, std::shared_ptr<CSR> csr,
    torch::Tensor lhs, torch::Tensor rhs, torch::Tensor out, int lhs_target, int rhs_target) {
  SWITCH_OP(op, Op, {
    SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, {
      cuda::SDDMMCsr<IdType, DType, Op, LhsTarget, RhsTarget>(
          bcast, csr, lhs, rhs, out);
    });
  });
}

/**
 * @brief CUDA implementation of g-SDDMM on Coo format.
 */
template <typename IdType, typename DType>
void SDDMMCoo(
    const std::string& op, const BcastOff& bcast, std::shared_ptr<COO> coo,
    torch::Tensor lhs, torch::Tensor rhs, torch::Tensor out, int lhs_target, int rhs_target) {
  SWITCH_OP(op, Op, {
    SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, {
      cuda::SDDMMCoo<IdType, DType, Op, LhsTarget, RhsTarget>(
          bcast, coo, lhs, rhs, out);
    });
  });
}

template void SDDMMCsr<int32_t, __half>(
    const std::string& op, const BcastOff& bcast, std::shared_ptr<CSR> csr,
    torch::Tensor lhs, torch::Tensor rhs, torch::Tensor out, int lhs_target, int rhs_target);
template void SDDMMCsr<int64_t, __half>(
    const std::string& op, const BcastOff& bcast, std::shared_ptr<CSR> csr,
    torch::Tensor lhs, torch::Tensor rhs, torch::Tensor out, int lhs_target, int rhs_target);

template void SDDMMCsr<int32_t, float>(
    const std::string& op, const BcastOff& bcast, std::shared_ptr<CSR> csr,
    torch::Tensor lhs, torch::Tensor rhs, torch::Tensor out, int lhs_target, int rhs_target);
template void SDDMMCsr<int64_t, float>(
    const std::string& op, const BcastOff& bcast, std::shared_ptr<CSR> csr,
    torch::Tensor lhs, torch::Tensor rhs, torch::Tensor out, int lhs_target, int rhs_target);
template void SDDMMCsr<int32_t, double>(
    const std::string& op, const BcastOff& bcast, std::shared_ptr<CSR> csr,
    torch::Tensor lhs, torch::Tensor rhs, torch::Tensor out, int lhs_target, int rhs_target);
template void SDDMMCsr<int64_t, double>(
    const std::string& op, const BcastOff& bcast, std::shared_ptr<CSR> csr,
    torch::Tensor lhs, torch::Tensor rhs, torch::Tensor out, int lhs_target, int rhs_target);

template void SDDMMCoo<int32_t, __half>(
    const std::string& op, const BcastOff& bcast, std::shared_ptr<COO> coo,
    torch::Tensor lhs, torch::Tensor rhs, torch::Tensor out, int lhs_target, int rhs_target);
template void SDDMMCoo<int64_t, __half>(
    const std::string& op, const BcastOff& bcast, std::shared_ptr<COO> coo,
    torch::Tensor lhs, torch::Tensor rhs, torch::Tensor out, int lhs_target, int rhs_target);

template void SDDMMCoo<int32_t, float>(
    const std::string& op, const BcastOff& bcast, std::shared_ptr<COO> coo,
    torch::Tensor lhs, torch::Tensor rhs, torch::Tensor out, int lhs_target, int rhs_target);
template void SDDMMCoo<int64_t, float>(
    const std::string& op, const BcastOff& bcast, std::shared_ptr<COO> coo,
    torch::Tensor lhs, torch::Tensor rhs, torch::Tensor out, int lhs_target, int rhs_target);
template void SDDMMCoo<int32_t, double>(
    const std::string& op, const BcastOff& bcast, std::shared_ptr<COO> coo,
    torch::Tensor lhs, torch::Tensor rhs, torch::Tensor out, int lhs_target, int rhs_target);
template void SDDMMCoo<int64_t, double>(
    const std::string& op, const BcastOff& bcast, std::shared_ptr<COO> coo,
    torch::Tensor lhs, torch::Tensor rhs, torch::Tensor out, int lhs_target, int rhs_target);

}  // namespace impl
}  // namespace gs