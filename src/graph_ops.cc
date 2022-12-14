#include "./graph_ops.h"

#include "bcast.h"
#include "cuda/fusion/column_row_slicing.h"
#include "cuda/fusion/random_walk.h"
#include "cuda/fusion/slice_sampling.h"
#include "cuda/graph_ops.h"
#include "cuda/sddmm.h"
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

std::pair<std::shared_ptr<CSC>, torch::Tensor> GraphCOO2DCSC(
    std::shared_ptr<COO> coo, bool COO2DCSC) {
  if (coo->row.device().type() == torch::kCUDA) {
    torch::Tensor indptr, indices, sort_index, val_ids;
    torch::optional<torch::Tensor> sorted_e_ids = torch::nullopt;
    if (COO2DCSC) {
      std::tie(indptr, indices, sort_index, val_ids) =
          impl::COO2DCSCCUDA(coo->row, coo->col);
    } else {
      std::tie(indptr, indices, sort_index, val_ids) =
          impl::COO2DCSCCUDA(coo->col, coo->row);
    }

    sorted_e_ids = coo->e_ids.has_value()
                       ? coo->e_ids.value().index({sort_index})
                       : sort_index;
    return {std::make_shared<CSC>(CSC{indptr, indices, sorted_e_ids}), val_ids};
  } else {
    LOG(FATAL) << "Not implemented warning";
    return {std::make_shared<CSC>(CSC{}), torch::Tensor()};
  }
}

std::shared_ptr<COO> GraphDCSC2COO(std::shared_ptr<CSC> csc, torch::Tensor ids,
                                   bool DCSC2COO) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor row, col;
    if (DCSC2COO) {
      std::tie(row, col) = impl::DCSC2COOCUDA(csc->indptr, csc->indices, ids);
    } else {
      std::tie(col, row) = impl::DCSC2COOCUDA(csc->indptr, csc->indices, ids);
    }
    return std::make_shared<COO>(COO{row, col, csc->e_ids});
  } else {
    LOG(FATAL) << "Not implemented warning";
    return std::make_shared<COO>(COO{});
  }
}

std::pair<std::shared_ptr<CSC>, torch::Tensor> FusedCSCColRowSlicing(
    std::shared_ptr<CSC> csc, torch::Tensor column_ids, torch::Tensor row_ids) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_indices, select_index;
    std::tie(sub_indptr, sub_indices, select_index) =
        impl::fusion::CSCColRowSlicingCUDA(csc->indptr, csc->indices,
                                           column_ids, row_ids);
    return {std::make_shared<CSC>(CSC{sub_indptr, sub_indices, torch::nullopt}),
            select_index};
  } else {
    LOG(FATAL) << "Not implemented warning";
    return {std::make_shared<CSC>(CSC{}), torch::Tensor()};
  }
}

std::pair<std::shared_ptr<_TMP>, torch::Tensor> CSCRowSlicing(
    std::shared_ptr<CSC> csc, torch::Tensor node_ids, bool with_coo) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, coo_col, coo_row, select_index;
    std::tie(sub_indptr, coo_col, coo_row, select_index) =
        impl::CSCRowSlicingCUDA(csc->indptr, csc->indices, node_ids, with_coo);
    return {std::make_shared<_TMP>(_TMP{sub_indptr, coo_col, coo_row}),
            select_index};
  } else {
    LOG(FATAL) << "Not implemented warning";
    return {std::make_shared<_TMP>(_TMP{}), torch::Tensor()};
  }
}

std::pair<std::shared_ptr<_TMP>, torch::Tensor> CSCColSlicing(
    std::shared_ptr<CSC> csc, torch::Tensor node_ids, bool with_coo) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, coo_col, coo_row, select_index;
    std::tie(sub_indptr, coo_col, coo_row, select_index) =
        impl::CSCColSlicingCUDA(csc->indptr, csc->indices, node_ids, with_coo);
    return {std::make_shared<_TMP>(_TMP{sub_indptr, coo_col, coo_row}),
            select_index};
  } else {
    LOG(FATAL) << "Not implemented warning";
    return {std::make_shared<_TMP>(_TMP{}), torch::Tensor()};
  }
}

std::pair<std::shared_ptr<_TMP>, torch::Tensor> DCSCColSlicing(
    std::shared_ptr<CSC> csc, torch::Tensor nid_map, torch::Tensor node_ids,
    bool with_coo) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, coo_col, coo_row, select_index;
    std::tie(sub_indptr, coo_col, coo_row, select_index) =
        impl::DCSCColSlicingCUDA(csc->indptr, csc->indices, nid_map, node_ids,
                                 with_coo);
    return {std::make_shared<_TMP>(_TMP{sub_indptr, coo_col, coo_row}),
            select_index};
  } else {
    LOG(FATAL) << "Not implemented warning";
    return {std::make_shared<_TMP>(_TMP{}), torch::Tensor()};
  }
}

// axis = 0 for column; reverse = 1 for row;
std::pair<std::shared_ptr<COO>, torch::Tensor> COOColSlicing(
    std::shared_ptr<COO> coo, torch::Tensor node_ids, int64_t axis) {
  if (coo->col.device().type() == torch::kCUDA) {
    torch::Tensor sub_coo_row, sub_coo_col, select_index;
    if (axis == 1)
      std::tie(sub_coo_row, sub_coo_col, select_index) =
          impl::COORowSlicingCUDA(coo->row, coo->col, node_ids);
    else
      std::tie(sub_coo_col, sub_coo_row, select_index) =
          impl::COORowSlicingCUDA(coo->col, coo->row, node_ids);
    return {
        std::make_shared<COO>(COO{sub_coo_row, sub_coo_col, torch::nullopt}),
        select_index};
  } else {
    LOG(FATAL) << "Not implemented warning";
    return {std::make_shared<COO>(COO{}), torch::Tensor()};
  }
}

std::pair<std::shared_ptr<_TMP>, torch::Tensor> CSCColSampling(
    std::shared_ptr<CSC> csc, int64_t fanout, bool replace, bool with_coo) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_coo_col, sub_indices, select_index;
    std::tie(sub_indptr, sub_coo_col, sub_indices, select_index) =
        impl::CSCColSamplingCUDA(csc->indptr, csc->indices, fanout, replace,
                                 with_coo);
    return {std::make_shared<_TMP>(_TMP{sub_indptr, sub_coo_col, sub_indices}),
            select_index};
  } else {
    LOG(FATAL) << "Not implemented warning";
    return {std::make_shared<_TMP>(_TMP{}), torch::Tensor()};
  }
}

std::pair<std::shared_ptr<_TMP>, torch::Tensor> CSCColSamplingProbs(
    std::shared_ptr<CSC> csc, torch::Tensor edge_probs, int64_t fanout,
    bool replace, bool with_coo) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_coo_col, sub_indices, select_index;
    std::tie(sub_indptr, sub_coo_col, sub_indices, select_index) =
        impl::CSCColSamplingProbsCUDA(csc->indptr, csc->indices, edge_probs,
                                      fanout, replace, with_coo);
    return {std::make_shared<_TMP>(_TMP{sub_indptr, sub_coo_col, sub_indices}),
            select_index};
  } else {
    LOG(FATAL) << "Not implemented warning";
    return {std::make_shared<_TMP>(_TMP{}), torch::Tensor()};
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

void CSCGraphSum(std::shared_ptr<CSC> csc, torch::optional<torch::Tensor> n_ids,
                 torch::Tensor data, torch::Tensor out_data, int64_t powk) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    impl::CSCSumCUDA(csc->indptr, csc->e_ids, n_ids, data, out_data, powk);
  } else {
    LOG(FATAL) << "Not implemented warning";
  }
}

void COOGraphSum(std::shared_ptr<COO> coo, torch::Tensor data,
                 torch::Tensor out_data, int64_t powk, int target_side) {
  if (coo->col.device().type() == torch::kCUDA) {
    auto target = (target_side == 0) ? coo->row : coo->col;
    impl::COOSumCUDA(target, coo->e_ids, data, out_data, powk);
  } else {
    LOG(FATAL) << "Not implemented warning";
  }
}

void CSCGraphDiv(std::shared_ptr<CSC> csc, torch::optional<torch::Tensor> n_ids,
                 torch::Tensor data, torch::Tensor divisor,
                 torch::Tensor out_data) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    const auto& bcast = CalcBcastOff("div", data, divisor);
    impl::SDDMMCSC("div", bcast, csc, n_ids, data, divisor, out_data, 1, 2);
  } else {
    LOG(FATAL) << "Not implemented warning";
  }
}

void COOGraphDiv(std::shared_ptr<COO> coo, torch::Tensor data,
                 torch::Tensor divisor, torch::Tensor out_data,
                 int target_side) {
  if (coo->col.device().type() == torch::kCUDA) {
    const auto& bcast = CalcBcastOff("div", data, divisor);
    int rhs_target = (target_side == 0) ? 0 : 2;
    impl::SDDMMCOO("div", bcast, coo, data, divisor, out_data, 1, rhs_target);
  } else {
    LOG(FATAL) << "Not implemented warning";
  }
}

void CSCGraphNormalize(std::shared_ptr<CSC> csc, torch::Tensor data,
                       torch::Tensor out_data) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    auto segmented_sum = torch::zeros(csc->indptr.numel() - 1, data.options());
    impl::CSCSumCUDA(csc->indptr, csc->e_ids, torch::nullopt, data,
                     segmented_sum, 1);
    const auto& bcast = CalcBcastOff("div", data, segmented_sum);
    impl::SDDMMCSC("div", bcast, csc, torch::nullopt, data, segmented_sum,
                   out_data, 1, 2);
  } else {
    LOG(FATAL) << "Not implemented warning";
  }
}

void COOGraphNormalize(std::shared_ptr<COO> coo, torch::Tensor data,
                       torch::Tensor out_data, int64_t side_len,
                       int target_side) {
  if (coo->col.device().type() == torch::kCUDA) {
    auto segmented_sum = torch::zeros(side_len, data.options());
    auto target = (target_side == 0) ? coo->row : coo->col;
    impl::COOSumCUDA(target, coo->e_ids, data, segmented_sum, 1);
    const auto& bcast = CalcBcastOff("div", data, segmented_sum);
    int rhs_target = (target_side == 0) ? 0 : 2;
    impl::SDDMMCOO("div", bcast, coo, data, segmented_sum, out_data, 1,
                   rhs_target);
  } else {
    LOG(FATAL) << "Not implemented warning";
  }
}

torch::Tensor FusedRandomWalk(std::shared_ptr<CSC> csc, torch::Tensor seeds,
                              int64_t walk_length) {
  torch::Tensor paths = impl::fusion::FusedRandomWalkCUDA(
      seeds, walk_length, csc->indices.data_ptr<int64_t>(),
      csc->indptr.data_ptr<int64_t>());
  return paths;
}

std::pair<std::shared_ptr<COO>, torch::Tensor> FullCSCColSlicing(
    std::shared_ptr<CSC> csc, torch::Tensor node_ids) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor coo_row, coo_col, select_index;
    std::tie(coo_row, coo_col, select_index) =
        impl::FullCSCColSlicingCUDA(csc->indptr, csc->indices, node_ids);
    return {std::make_shared<COO>(COO{coo_row, coo_col, torch::nullopt}),
            select_index};
  } else {
    LOG(FATAL) << "Not implemented warning";
    return {std::make_shared<COO>(COO{}), torch::Tensor()};
  }
}

std::pair<std::shared_ptr<COO>, torch::Tensor> FullCSCRowSlicing(
    std::shared_ptr<CSC> csc, torch::Tensor node_ids) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor coo_row, coo_col, select_index;
    std::tie(coo_row, coo_col, select_index) =
        impl::FullCSCRowSlicingCUDA(csc->indptr, csc->indices, node_ids);
    return {std::make_shared<COO>(COO{coo_row, coo_col, torch::nullopt}),
            select_index};
  } else {
    LOG(FATAL) << "Not implemented warning";
    return {std::make_shared<COO>(COO{}), torch::Tensor()};
  }
}

// axis = 0 for column; reverse = 1 for row;
std::pair<std::shared_ptr<COO>, torch::Tensor> FullCOOColSlicing(
    std::shared_ptr<COO> coo, torch::Tensor node_ids, int64_t axis) {
  if (coo->row.device().type() == torch::kCUDA) {
    torch::Tensor coo_row, coo_col, select_index;
    if (axis == 1)
      std::tie(coo_row, coo_col, select_index) =
          impl::FullCOORowSlicingCUDA(coo->row, coo->col, node_ids);
    else
      std::tie(coo_row, coo_col, select_index) =
          impl::FullCOORowSlicingCUDA(coo->col, coo->row, node_ids);
    return {std::make_shared<COO>(COO{coo_row, coo_col, torch::nullopt}),
            select_index};
  } else {
    LOG(FATAL) << "Not implemented warning";
    return {std::make_shared<COO>(COO{}), torch::Tensor()};
  }
}

std::pair<std::shared_ptr<COO>, torch::Tensor> FullCSCColSampling(
    std::shared_ptr<CSC> csc, int64_t fanout, bool replace) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor coo_row, coo_col, select_index;
    std::tie(coo_row, coo_col, select_index) = impl::FullCSCColSamplingCUDA(
        csc->indptr, csc->indices, fanout, replace);
    return {std::make_shared<COO>(COO{coo_row, coo_col, torch::nullopt}),
            select_index};
  } else {
    LOG(FATAL) << "Not implemented warning";
    return {std::make_shared<COO>(COO{}), torch::Tensor()};
  }
}

std::pair<std::shared_ptr<COO>, torch::Tensor> FullCSCColSamplingProbs(
    std::shared_ptr<CSC> csc, torch::Tensor edge_probs, int64_t fanout,
    bool replace) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor coo_row, coo_col, select_index;
    std::tie(coo_row, coo_col, select_index) =
        impl::FullCSCColSamplingProbsCUDA(csc->indptr, csc->indices, edge_probs,
                                          fanout, replace);
    return {std::make_shared<COO>(COO{coo_row, coo_col, torch::nullopt}),
            select_index};
  } else {
    LOG(FATAL) << "Not implemented warning";
    return {std::make_shared<COO>(COO{}), torch::Tensor()};
  }
}

}  // namespace gs
