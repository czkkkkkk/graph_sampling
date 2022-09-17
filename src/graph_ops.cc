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
    torch::Tensor indptr, indices;
    torch::optional<torch::Tensor> e_ids;
    std::tie(indptr, indices, e_ids) =
        impl::GraphCOO2CSRCUDA(coo->row, coo->col, coo->e_ids, num_rows);
    return std::make_shared<CSR>(
        CSR{indptr, indices, e_ids});
  } else {
    LOG(FATAL) << "Not implemented warning";
    return std::make_shared<CSR>(CSR{});
  }
}

std::shared_ptr<CSC> GraphCOO2CSC(std::shared_ptr<COO> coo, int64_t num_cols) {
  if (coo->row.device().type() == torch::kCUDA) {
    torch::Tensor indptr, indices;
    torch::optional<torch::Tensor> e_ids;
    std::tie(indptr, indices, e_ids) =
        impl::GraphCOO2CSRCUDA(coo->col, coo->row, coo->e_ids, num_cols);
    return std::make_shared<CSC>(
        CSC{indptr, indices, e_ids});
  } else {
    LOG(FATAL) << "Not implemented warning";
    return std::make_shared<CSC>(CSC{});
  }
}

std::shared_ptr<CSC> CSCColumnwiseSlicing(std::shared_ptr<CSC> csc,
                                          torch::Tensor column_ids) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_indices, sub_e_ids, e_ids;
    e_ids = (csc->e_ids.has_value()) ? csc->e_ids.value()
                                     : torch::arange(0, csc->indices.numel(),
                                                     csc->indices.options());
    std::tie(sub_indptr, sub_indices, sub_e_ids) =
        impl::CSCColumnwiseSlicingCUDA(csc->indptr, csc->indices, e_ids,
                                       column_ids);
    return std::make_shared<CSC>(
        CSC{sub_indptr, sub_indices, torch::make_optional(sub_e_ids)});
  } else {
    LOG(FATAL) << "Not implemented warning";
    return std::make_shared<CSC>(CSC{});
  }
}

// @todo slicing with e_ids
std::shared_ptr<CSC> CSCRowwiseSlicing(std::shared_ptr<CSC> csc,
                                       torch::Tensor row_ids) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_indices;
    std::tie(sub_indptr, sub_indices) =
        impl::CSCRowwiseSlicingCUDA(csc->indptr, csc->indices, row_ids);
    return std::make_shared<CSC>(CSC{sub_indptr, sub_indices, torch::nullopt});
  } else {
    std::cerr << "Not implemented warning";
    return std::make_shared<CSC>(CSC{});
  }
}

std::shared_ptr<CSR> CSRRowwiseSlicing(std::shared_ptr<CSR> csr,
                                       torch::Tensor row_ids) {
  if (csr->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_indices, sub_e_ids, e_ids;
    e_ids = (csr->e_ids.has_value()) ? csr->e_ids.value()
                                     : torch::arange(0, csr->indices.numel(),
                                                     csr->indices.options());
    std::tie(sub_indptr, sub_indices, sub_e_ids) =
        impl::CSCColumnwiseSlicingCUDA(csr->indptr, csr->indices, e_ids,
                                       row_ids);
    return std::make_shared<CSR>(CSR{sub_indptr, sub_indices, sub_e_ids});
  } else {
    LOG(FATAL) << "Not implemented warning";
    return std::make_shared<CSR>(CSR{});
  }
}

std::shared_ptr<CSC> CSCColumnwiseSampling(std::shared_ptr<CSC> csc,
                                           int64_t fanout, bool replace) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_indices, sub_e_ids, e_ids;
    e_ids = (csc->e_ids.has_value()) ? csc->e_ids.value()
                                     : torch::arange(0, csc->indices.numel(),
                                                     csc->indices.options());
    std::tie(sub_indptr, sub_indices, sub_e_ids) =
        impl::CSCColumnwiseSamplingCUDA(csc->indptr, csc->indices, e_ids,
                                        fanout, replace);
    return std::make_shared<CSC>(CSC{sub_indptr, sub_indices, sub_e_ids});
  } else {
    LOG(FATAL) << "Not implemented warning";
    return std::make_shared<CSC>(CSC{});
  }
}

std::shared_ptr<CSC> CSCColumnwiseFusedSlicingAndSampling(
    std::shared_ptr<CSC> csc, torch::Tensor column_ids, int64_t fanout,
    bool replace) {
  if (csc->indptr.device().type() == torch::kCUDA) {
    torch::Tensor sub_indptr, sub_indices, sub_e_ids, e_ids;
    e_ids = (csc->e_ids.has_value()) ? csc->e_ids.value()
                                     : torch::arange(0, csc->indices.numel(),
                                                     csc->indices.options());
    std::tie(sub_indptr, sub_indices, sub_e_ids) =
        impl::CSCColumnwiseFusedSlicingAndSamplingCUDA(
            csc->indptr, csc->indices, e_ids, column_ids, fanout, replace);
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
