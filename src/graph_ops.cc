#include "./graph_ops.h"

#include "cuda/graph_ops.h"

namespace gs {

std::shared_ptr<COO> GraphCSC2COO(std::shared_ptr<CSC> csc) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor row, col;
    std::tie(row, col) = impl::GraphCSC2COOCUDA(csc->indptr, csc->indices);
    return std::make_shared<COO>(COO{row, col, csc->e_ids});
  } else {
    LOG(FATAL) << "Not implemented warning";
    return std::make_shared<COO>(COO{});
  }
}

std::shared_ptr<COO> GraphCSR2COO(std::shared_ptr<CSR> csr) {
  if (csr->indptr.device().type() == torch::kCUDA) {
    torch::Tensor row, col;
    std::tie(col, row) = impl::GraphCSC2COOCUDA(csr->indptr, csr->indices);
    return std::make_shared<COO>(COO{row, col, csr->e_ids});
  } else {
    LOG(FATAL) << "Not implemented warning";
    return std::make_shared<COO>(COO{});
  }
}

std::shared_ptr<CSR> GraphCOO2CSR(std::shared_ptr<COO> coo, int64_t num_rows) {
  if (coo->row.device().type() == torch::kCUDA) {
    torch::Tensor indptr, indices, sort_index;
    torch::optional<torch::Tensor> sorted_e_ids = torch::nullopt;
    std::tie(indptr, indices, sort_index) =
        impl::GraphCOO2CSRCUDA(coo->row, coo->col, num_rows);
    if (coo->e_ids.has_value()) {
      sorted_e_ids =
          torch::make_optional(coo->e_ids.value().index({sort_index}));
    }
    return std::make_shared<CSR>(CSR{indptr, indices, sorted_e_ids});
  } else {
    LOG(FATAL) << "Not implemented warning";
    return std::make_shared<CSR>(CSR{});
  }
}

std::shared_ptr<CSC> GraphCOO2CSC(std::shared_ptr<COO> coo, int64_t num_cols) {
  if (coo->row.device().type() == torch::kCUDA) {
    torch::Tensor indptr, indices, sort_index;
    torch::optional<torch::Tensor> sorted_e_ids = torch::nullopt;
    std::tie(indptr, indices, sort_index) =
        impl::GraphCOO2CSRCUDA(coo->col, coo->row, num_cols);
    if (coo->e_ids.has_value()) {
      sorted_e_ids =
          torch::make_optional(coo->e_ids.value().index({sort_index}));
    }
    return std::make_shared<CSC>(CSC{indptr, indices, sorted_e_ids});
  } else {
    LOG(FATAL) << "Not implemented warning";
    return std::make_shared<CSC>(CSC{});
  }
}

std::shared_ptr<CSC> CSCColumnwiseSlicing(std::shared_ptr<CSC> csc,
                                          torch::Tensor column_ids,
                                          torch::optional<torch::Tensor> data) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_indices, select_index;
    torch::optional<torch::Tensor> sub_e_ids = torch::nullopt;
    std::tie(sub_indptr, sub_indices, select_index) =
        impl::CSCColumnwiseSlicingCUDA(csc->indptr, csc->indices, column_ids);
    if (data.has_value()) {
      if (csc->e_ids.has_value()) {
        sub_e_ids = csc->e_ids.value().index({select_index});
      } else {
        sub_e_ids = select_index;
      }
    }
    return std::make_shared<CSC>(CSC{sub_indptr, sub_indices, sub_e_ids});
  } else {
    LOG(FATAL) << "Not implemented warning";
    return std::make_shared<CSC>(CSC{});
  }
}

std::shared_ptr<CSC> CSCRowwiseSlicing(std::shared_ptr<CSC> csc,
                                       torch::Tensor row_ids,
                                       torch::optional<torch::Tensor> data) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_indices, select_index;
    torch::optional<torch::Tensor> sub_e_ids = torch::nullopt;
    std::tie(sub_indptr, sub_indices, select_index) =
        impl::CSCRowwiseSlicingCUDA(csc->indptr, csc->indices, row_ids);
    if (data.has_value()) {
      if (csc->e_ids.has_value()) {
        sub_e_ids = csc->e_ids.value().index({select_index});
      } else {
        sub_e_ids = select_index;
      }
    }
    return std::make_shared<CSC>(CSC{sub_indptr, sub_indices, sub_e_ids});
  } else {
    std::cerr << "Not implemented warning";
    return std::make_shared<CSC>(CSC{});
  }
}

std::shared_ptr<CSR> CSRRowwiseSlicing(std::shared_ptr<CSR> csr,
                                       torch::Tensor row_ids,
                                       torch::optional<torch::Tensor> data) {
  if (csr->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_indices, select_index;
    torch::optional<torch::Tensor> sub_e_ids = torch::nullopt;
    std::tie(sub_indptr, sub_indices, select_index) =
        impl::CSCColumnwiseSlicingCUDA(csr->indptr, csr->indices, row_ids);
    if (data.has_value()) {
      if (csr->e_ids.has_value()) {
        sub_e_ids = csr->e_ids.value().index({select_index});
      } else {
        sub_e_ids = select_index;
      }
    }
    return std::make_shared<CSR>(CSR{sub_indptr, sub_indices, sub_e_ids});
  } else {
    LOG(FATAL) << "Not implemented warning";
    return std::make_shared<CSR>(CSR{});
  }
}

std::shared_ptr<CSC> CSCColumnwiseSampling(
    std::shared_ptr<CSC> csc, int64_t fanout, bool replace,
    torch::optional<torch::Tensor> data) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_indices, select_index;
    torch::optional<torch::Tensor> sub_e_ids = torch::nullopt;
    std::tie(sub_indptr, sub_indices, select_index) =
        impl::CSCColumnwiseSamplingCUDA(csc->indptr, csc->indices, fanout,
                                        replace);
    if (data.has_value()) {
      if (csc->e_ids.has_value()) {
        sub_e_ids = csc->e_ids.value().index({select_index});
      } else {
        sub_e_ids = select_index;
      }
    }
    return std::make_shared<CSC>(CSC{sub_indptr, sub_indices, sub_e_ids});
  } else {
    LOG(FATAL) << "Not implemented warning";
    return std::make_shared<CSC>(CSC{});
  }
}

std::shared_ptr<CSC> CSCColumnwiseFusedSlicingAndSampling(
    std::shared_ptr<CSC> csc, torch::Tensor column_ids, int64_t fanout,
    bool replace, torch::optional<torch::Tensor> data) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_indices, select_index;
    torch::optional<torch::Tensor> sub_e_ids = torch::nullopt;
    std::tie(sub_indptr, sub_indices, select_index) =
        impl::CSCColumnwiseFusedSlicingAndSamplingCUDA(
            csc->indptr, csc->indices, column_ids, fanout, replace);
    if (data.has_value()) {
      if (csc->e_ids.has_value()) {
        sub_e_ids = csc->e_ids.value().index({select_index});
      } else {
        sub_e_ids = select_index;
      }
    }
    return std::make_shared<CSC>(CSC{sub_indptr, sub_indices, sub_e_ids});
  } else {
    LOG(FATAL) << "Not implemented warning";
    return std::make_shared<CSC>(CSC{});
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

// @todo Fix for new storage format
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> GraphRelabel(
    torch::Tensor col_ids, torch::Tensor indptr, torch::Tensor indices) {
  torch::Tensor frontier, relabeled_indices, relabeled_indptr;
  std::tie(frontier, relabeled_indices) = impl::RelabelCUDA(col_ids, indices);
  relabeled_indptr = indptr.clone();
  return std::make_tuple(frontier, relabeled_indptr, relabeled_indices);
}

torch::Tensor GraphSum(torch::Tensor indptr, torch::Tensor e_ids,
                       torch::Tensor data) {
  if (indptr.device().type() == torch::kCUDA) {
    return impl::GraphSumCUDA(indptr, e_ids, data);
  } else {
    LOG(FATAL) << "Not implemented warning";
    return torch::Tensor();
  }
}

torch::Tensor GraphDiv(torch::Tensor indptr, torch::Tensor e_ids,
                       torch::Tensor data, torch::Tensor divisor) {
  if (indptr.device().type() == torch::kCUDA) {
    return impl::GraphDivCUDA(indptr, e_ids, data, divisor);
  } else {
    LOG(FATAL) << "Not implemented warning";
    return torch::Tensor();
  }
}

torch::Tensor GraphL2Norm(torch::Tensor indptr, torch::Tensor e_ids,
                          torch::Tensor data) {
  if (indptr.device().type() == torch::kCUDA) {
    return impl::GraphL2NormCUDA(indptr, e_ids, data);
  } else {
    LOG(FATAL) << "Not implemented warning";
    return torch::Tensor();
  }
}

torch::Tensor GraphNormalize(torch::Tensor indptr, torch::Tensor e_ids,
                             torch::Tensor data) {
  if (indptr.device().type() == torch::kCUDA) {
    return impl::GraphNormalizeCUDA(indptr, e_ids, data);
  } else {
    LOG(FATAL) << "Not implemented warning";
    return torch::Tensor();
  }
}

}  // namespace gs
