#include "./graph_ops.h"

#include "cuda/graph_ops.h"
#include "cuda/heterograph_ops.h"

namespace gs {

std::shared_ptr<CSC> CSCColumnwiseSlicing(std::shared_ptr<CSC> csc,
                                          torch::Tensor column_ids) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_indices;
    std::tie(sub_indptr, sub_indices) =
        impl::CSCColumnwiseSlicingCUDA(csc->indptr, csc->indices, column_ids);
    return std::make_shared<CSC>(CSC{column_ids, sub_indptr, sub_indices});
  } else {
    LOG(FATAL) << "Not implemented warning";
    return std::make_shared<CSC>(CSC{});
  }
}

std::shared_ptr<CSC> CSCRowwiseSlicing(std::shared_ptr<CSC> csc,
                                       torch::Tensor row_ids) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_indices;
    std::tie(sub_indptr, sub_indices) =
        impl::CSCRowwiseSlicingCUDA(csc->indptr, csc->indices, row_ids);
    return std::make_shared<CSC>(CSC{csc->col_ids, sub_indptr, sub_indices});
  } else {
    std::cerr << "Not implemented warning";
  }
}

torch::Tensor TensorUnique(torch::Tensor node_ids) {
  if (node_ids.device().type() == torch::kCUDA) {
    return impl::TensorUniqueCUDA(node_ids);
  } else {
    LOG(FATAL) << "Not implemented warning";
    return torch::Tensor();
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GraphRelabel(
    torch::Tensor col_ids, torch::Tensor indptr, torch::Tensor indices) {
  torch::Tensor frontier, relabeled_indices, relabeled_indptr;
  std::tie(frontier, relabeled_indices) = impl::RelabelCUDA(col_ids, indices);
  relabeled_indptr = indptr.clone();
  return std::make_tuple(frontier, relabeled_indptr, relabeled_indices);
}

std::shared_ptr<CSC> CSCColumnwiseSampling(std::shared_ptr<CSC> csc,
                                           int64_t fanout, bool replace) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_indices;
    std::tie(sub_indptr, sub_indices) = impl::CSCColumnwiseSamplingCUDA(
        csc->indptr, csc->indices, fanout, replace);
    return std::make_shared<CSC>(CSC{csc->col_ids, sub_indptr, sub_indices});
  } else {
    LOG(FATAL) << "Not implemented warning";
    return std::make_shared<CSC>(CSC{});
  }
}

std::shared_ptr<CSC> CSCColumnwiseFusedSlicingAndSampling(
    std::shared_ptr<CSC> csc, torch::Tensor column_ids, int64_t fanout,
    bool replace) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_indices;
    if (fanout == 1 && replace) {
      std::tie(sub_indptr, sub_indices) =
          impl::CSCColumnwiseSamplingOneKeepDimCUDA(csc->indptr, csc->indices,
                                                    column_ids);
      return std::make_shared<CSC>(CSC{column_ids, sub_indptr, sub_indices});
    } else {
      std::tie(sub_indptr, sub_indices) =
          impl::CSCColumnwiseFusedSlicingAndSamplingCUDA(
              csc->indptr, csc->indices, column_ids, fanout, replace);
      return std::make_shared<CSC>(CSC{column_ids, sub_indptr, sub_indices});
    }
  } else {
    LOG(FATAL) << "Not implemented warning";
    return std::make_shared<CSC>(CSC{});
  }
}

torch::Tensor RandomWalkFused(std::shared_ptr<CSC> csc, torch::Tensor seeds,
                              int64_t walk_length) {
  torch::Tensor paths = impl::RandomWalkFusedCUDA(
      seeds, walk_length, csc->indices.data_ptr<int64_t>(),
      csc->indptr.data_ptr<int64_t>());
  return paths;
}

}  // namespace gs
