#ifndef GS_KERNEL_H_
#define GS_KERNEL_H_

#include <torch/script.h>
#include "./graph.h"

namespace gs {
void SDDMM(const std::string& op, c10::intrusive_ptr<Graph> graph,
           torch::Tensor lhs, torch::Tensor rhs, torch::Tensor out,
           int lhs_target, int rhs_target);
}

#endif