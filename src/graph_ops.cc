#include "./graph_ops.h"

#include <iostream>

#include "cuda/graph_ops.h"

namespace gs {

std::shared_ptr<CSC> CSCColumnwiseSlicing(std::shared_ptr<CSC> csc,
                                          torch::Tensor column_ids) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_indices;
    std::tie(sub_indptr, sub_indices) =
        impl::CSCColumnwiseSlicingCUDA(csc->indptr, csc->indices, column_ids);
    return std::make_shared<CSC>(CSC{sub_indptr, sub_indices});
  } else {
    std::cerr << "Not implemented warning";
  }
}

std::shared_ptr<CSC> CSCColumnwiseSampling(std::shared_ptr<CSC> csc,
                                           int64_t fanout, bool replace) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_indices;
    std::tie(sub_indptr, sub_indices) = impl::CSCColumnwiseSamplingCUDA(
        csc->indptr, csc->indices, fanout, replace);
    return std::make_shared<CSC>(CSC{sub_indptr, sub_indices});
  } else {
    std::cerr << "Not implemented warning";
  }
}

std::shared_ptr<CSC> CSCColumnwiseFusedSlicingAndSampling(
    std::shared_ptr<CSC> csc, torch::Tensor column_ids, int64_t fanout,
    bool replace) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_indices;
    std::tie(sub_indptr, sub_indices) =
        impl::CSCColumnwiseFusedSlicingAndSamplingCUDA(
            csc->indptr, csc->indices, column_ids, fanout, replace);
    return std::make_shared<CSC>(CSC{sub_indptr, sub_indices});
  } else {
    std::cerr << "Not implemented warning";
  }
}

}  // namespace gs