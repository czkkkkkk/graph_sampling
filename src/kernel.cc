#include "./kernel.h"
#include "./bcast.h"
#include "./cuda/sddmm.h"

namespace gs {

/*! \brief Generalized Sampled Dense-Dense Matrix Multiplication. */
void SDDMM(const std::string& op, c10::intrusive_ptr<Graph> graph,
           torch::Tensor lhs, torch::Tensor rhs, torch::Tensor out,
           int lhs_target, int rhs_target) {
  const auto& bcast = CalcBcastOff(op, lhs, rhs);
  auto num_rows = graph->GetNumRows(), num_cols = graph->GetNumCols();
  if (graph->GetCSR() != nullptr) {
    impl::SDDMMCSR(op, bcast, graph->GetCSR(), lhs, rhs, out, lhs_target,
                   rhs_target, num_rows, num_cols);
  } else if (graph->GetCOO() != nullptr) {
    impl::SDDMMCOO(op, bcast, graph->GetCOO(), lhs, rhs, out, lhs_target,
                   rhs_target, num_rows, num_cols);
  } else {
    LOG(FATAL) << "SDDMM only supports CSR and COO formats";
  }
}
}  // namespace gs