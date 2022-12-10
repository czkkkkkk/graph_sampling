#include "./graph_ops.h"

#include "cuda/fusion/column_row_slicing.h"
#include "cuda/fusion/random_walk.h"
#include "cuda/fusion/slice_sampling.h"
#include "cuda/graph_ops.h"
#include "cuda/tensor_ops.h"

namespace gs {

std::shared_ptr<COO> GraphCSC2COO(std::shared_ptr<CSC> csc, bool CSC2COO) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor row, col;
    if (CSC2COO) {
      std::tie(row, col) = impl::CSC2COOCUDA(csc->indptr, csc->indices);
    } else {
      std::tie(col, row) = impl::CSC2COOCUDA(csc->indptr, csc->indices);
    }
    return std::make_shared<COO>(COO{row, col, csc->e_ids});
  } else {
    LOG(FATAL) << "Not implemented warning";
    return std::make_shared<COO>(COO{});
  }
}

std::shared_ptr<CSC> GraphCOO2CSC(std::shared_ptr<COO> coo, int64_t num_items,
                                  bool COO2CSC) {
  if (coo->row.device().type() == torch::kCUDA) {
    torch::Tensor indptr, indices, sort_index;
    torch::optional<torch::Tensor> sorted_e_ids = torch::nullopt;
    if (COO2CSC) {
      std::tie(indptr, indices, sort_index) =
          impl::COO2CSCCUDA(coo->row, coo->col, num_items);
    } else {
      std::tie(indptr, indices, sort_index) =
          impl::COO2CSCCUDA(coo->col, coo->row, num_items);
    }

    sorted_e_ids = coo->e_ids.has_value()
                       ? coo->e_ids.value().index({sort_index})
                       : sort_index;
    return std::make_shared<CSC>(CSC{indptr, indices, sorted_e_ids});
  } else {
    LOG(FATAL) << "Not implemented warning";
    return std::make_shared<CSC>(CSC{});
  }
}

std::pair<std::shared_ptr<CSC>, torch::Tensor> FusedCSCColRowSlicing(
    std::shared_ptr<CSC> csc, torch::Tensor seeds) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_indices, select_index;
    std::tie(sub_indptr, sub_indices, select_index) =
        impl::fusion::CSCColRowSlicingCUDA(csc->indptr, csc->indices, seeds);
    return {std::make_shared<CSC>(CSC{sub_indptr, sub_indices, torch::nullopt}),
            select_index};
  } else {
    LOG(FATAL) << "Not implemented warning";
    return {std::make_shared<CSC>(CSC{}), torch::Tensor()};
  }
}

std::pair<std::shared_ptr<CSC>, torch::Tensor> CSCRowSlicing(
    std::shared_ptr<CSC> csc, torch::Tensor node_ids) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_indices, select_index;
    std::tie(sub_indptr, sub_indices, select_index) =
        impl::OnIndicesSlicingCUDA(csc->indptr, csc->indices, node_ids);
    return {std::make_shared<CSC>(CSC{sub_indptr, sub_indices, torch::nullopt}),
            select_index};
  } else {
    LOG(FATAL) << "Not implemented warning";
    return {std::make_shared<CSC>(CSC{}), torch::Tensor()};
  }
}

std::pair<std::shared_ptr<CSC>, torch::Tensor> CSCColSlicing(
    std::shared_ptr<CSC> csc, torch::Tensor node_ids) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_indices, select_index;
    std::tie(sub_indptr, sub_indices, select_index) =
        impl::OnIndptrSlicingCUDA(csc->indptr, csc->indices, node_ids);
    return {std::make_shared<CSC>(CSC{sub_indptr, sub_indices, torch::nullopt}),
            select_index};
  } else {
    LOG(FATAL) << "Not implemented warning";
    return {std::make_shared<CSC>(CSC{}), torch::Tensor()};
  }
}

std::pair<std::shared_ptr<CSC>, torch::Tensor> CSCColSampling(
    std::shared_ptr<CSC> csc, int64_t fanout, bool replace) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_indices, select_index;
    std::tie(sub_indptr, sub_indices, select_index) =
        impl::CSCColSamplingCUDA(csc->indptr, csc->indices, fanout, replace);
    return {std::make_shared<CSC>(CSC{sub_indptr, sub_indices, torch::nullopt}),
            select_index};
  } else {
    LOG(FATAL) << "Not implemented warning";
    return {std::make_shared<CSC>(CSC{}), torch::Tensor()};
  }
}

std::pair<std::shared_ptr<CSC>, torch::Tensor> CSCColSamplingProbs(
    std::shared_ptr<CSC> csc, torch::Tensor edge_probs, int64_t fanout,
    bool replace) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_indices, select_index;
    std::tie(sub_indptr, sub_indices, select_index) =
        impl::CSCColSamplingProbsCUDA(csc->indptr, csc->indices, edge_probs,
                                      fanout, replace);
    return {std::make_shared<CSC>(CSC{sub_indptr, sub_indices, torch::nullopt}),
            select_index};
  } else {
    LOG(FATAL) << "Not implemented warning";
    return {std::make_shared<CSC>(CSC{}), torch::Tensor()};
  }
}

std::pair<std::shared_ptr<CSC>, torch::Tensor> FusedCSCColSlicingAndSampling(
    std::shared_ptr<CSC> csc, torch::Tensor node_ids, int64_t fanout,
    bool replace) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_indices, select_index;
    if (fanout == 1 && replace) {
      std::tie(sub_indptr, sub_indices, select_index) =
          impl::fusion::FusedCSCColSlicingSamplingOneKeepDimCUDA(
              csc->indptr, csc->indices, node_ids);
      return {
          std::make_shared<CSC>(CSC{sub_indptr, sub_indices, torch::nullopt}),
          select_index};
    } else {
      std::tie(sub_indptr, sub_indices, select_index) =
          impl::fusion::FusedCSCColSlicingSamplingCUDA(
              csc->indptr, csc->indices, node_ids, fanout, replace);
      return {
          std::make_shared<CSC>(CSC{sub_indptr, sub_indices, torch::nullopt}),
          select_index};
    }
  } else {
    LOG(FATAL) << "Not implemented warning";
    return {std::make_shared<CSC>(CSC{}), torch::Tensor()};
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

// BatchTensorRelabel leverages vector<Tensor> mapping_tensors to create the
// hashmap which stores the mapping. Then, it will do relabel operation for
// tensor in to_be_relabeled_tensors with the hashmap. It return {unique_tensor,
// {tensor1_after_relabeled, tensor2_after_relabeled, ...}}.
std::tuple<torch::Tensor, std::vector<torch::Tensor>> BatchTensorRelabel(
    std::vector<torch::Tensor> mapping_tensors,
    std::vector<torch::Tensor> to_be_relabeled_tensors) {
  torch::Tensor frontier;
  std::vector<torch::Tensor> relabel_result;
  std::tie(frontier, relabel_result) =
      impl::RelabelCUDA(mapping_tensors, to_be_relabeled_tensors);
  return std::make_tuple(frontier, relabel_result);
}

torch::Tensor GraphSum(std::shared_ptr<CSC> csc,
                       torch::optional<torch::Tensor> data, int64_t powk) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    return impl::CSCSumCUDA(csc->indptr, csc->e_ids, data, powk);
  } else {
    LOG(FATAL) << "Not implemented warning";
    return torch::Tensor();
  }
}

torch::Tensor GraphDiv(std::shared_ptr<CSC> csc,
                       torch::optional<torch::Tensor> data,
                       torch::Tensor divisor) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    return impl::CSCDivCUDA(csc->indptr, csc->e_ids, data, divisor);
  } else {
    LOG(FATAL) << "Not implemented warning";
    return torch::Tensor();
  }
}

torch::Tensor GraphNormalize(std::shared_ptr<CSC> csc,
                             torch::optional<torch::Tensor> data) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    return impl::CSCNormalizeCUDA(csc->indptr, csc->e_ids, data);
  } else {
    LOG(FATAL) << "Not implemented warning";
    return torch::Tensor();
  }
}

torch::Tensor FusedRandomWalk(std::shared_ptr<CSC> csc, torch::Tensor seeds,
                              int64_t walk_length) {
  torch::Tensor paths = impl::fusion::FusedRandomWalkCUDA(
      seeds, walk_length, csc->indices.data_ptr<int64_t>(),
      csc->indptr.data_ptr<int64_t>());
  return paths;
}

}  // namespace gs
