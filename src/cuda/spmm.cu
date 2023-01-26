#include "./spmm.h"

#include "./cuda_common.h"
#include "./macro.h"
#include "./selector.h"
#include "./utils.h"

namespace gs {
namespace impl {
/**
 * @brief CUDA implementation of g-SpMM on CSC format.
 */
std::pair<torch::Tensor, torch::Tensor> SpMMCSC(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const std::shared_ptr<CSC> csc, torch::optional<torch::Tensor> n_ids,
    torch::Tensor ufeat, torch::Tensor efeat, torch::Tensor out,
    std::vector<torch::Tensor> out_aux) {}

/**
 * @brief CUDA implementation of g-SpMM on COO format.
 */
std::pair<torch::Tensor, torch::Tensor> SpMMCOO(
    const std::string& op, const std::string& reduce, const BcastOff& bcast,
    const std::shared_ptr<COO> coo, torch::Tensor ufeat, torch::Tensor efeat,
    torch::Tensor out, std::vector<torch::Tensor> out_aux) {
  //   if (reduce == "sum") {
  //     SWITCH_BITS(coo->row.scalar_type(), IdType, {
  //       SWITCH_BITS(out.scalar_type(), DType, {
  //         SWITCH_OP(op, Op, {
  //           SpMMCoo<IdType, DType, Op,
  //                         impl::reduce::Sum<IdType, DType, true> >(
  //               bcast, coo, ufeat, efeat, out, out_aux[0], out_aux[1]);
  //         });
  //       });
  //     });
  //   } else if (reduce == "max") {
  //     SWITCH_OP(op, Op, {
  //       SpMMCoo<IdType, DType, Op, cuda::reduce::Max<IdType, DType, true> >(
  //           bcast, coo, ufeat, efeat, out, out_aux[0], out_aux[1]);
  //     });
  //   } else if (reduce == "min") {
  //     SWITCH_OP(op, Op, {
  //       SpMMCoo<IdType, DType, Op, cuda::reduce::Min<IdType, DType, true> >(
  //           bcast, coo, ufeat, efeat, out, out_aux[0], out_aux[1]);
  //     });
  //   } else {
  //     LOG(FATAL) << "Not implemented warning";
  //   }
}
}  // namespace impl
}  // namespace gs