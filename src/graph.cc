#include "./graph.h"

#include <sstream>
#include "bcast.h"
#include "cuda/fusion/column_row_slicing.h"
#include "cuda/fusion/edge_map_reduce.h"
#include "cuda/graph_ops.h"
#include "cuda/sddmm.h"
#include "cuda/tensor_ops.h"
#include "graph_ops.h"

namespace gs {

void Graph::CreateSparseFormat(int64_t format) {
  if (format == _COO) {
    if (coo_ != nullptr) return;
    if (csc_ != nullptr) {
      SetCOO(GraphCSC2COO(csc_, true));
    } else {
      SetCOO(GraphCSC2COO(csr_, false));
    }
  } else if (format == _CSC) {
    if (csc_ != nullptr) return;
    if (coo_ != nullptr)
      SetCSC(GraphCOO2CSC(coo_, num_cols_, true));
    else {
      SetCOO(GraphCSC2COO(csr_, false));
      SetCSC(GraphCOO2CSC(coo_, num_cols_, true));
    }
  } else if (format == _CSR) {
    if (csr_ != nullptr) return;
    if (coo_ != nullptr)
      SetCSR(GraphCOO2CSC(coo_, num_rows_, false));
    else {
      SetCOO(GraphCSC2COO(csc_, true));
      SetCSR(GraphCOO2CSC(coo_, num_rows_, false));
    }
  } else {
    LOG(FATAL) << "Unsupported sparse format!";
  }
}

torch::Tensor Graph::GetCSCIndptr() {
  CreateSparseFormat(_CSC);
  return csc_->indptr;
}
torch::Tensor Graph::GetCSCIndices() {
  CreateSparseFormat(_CSC);
  return csc_->indices;
}
torch::Tensor Graph::GetCSCEids() {
  CreateSparseFormat(_CSC);
  return csc_->e_ids.has_value() ? csc_->e_ids.value() : torch::Tensor();
}
torch::Tensor Graph::GetCOORows() {
  CreateSparseFormat(_COO);
  return coo_->row;
}
torch::Tensor Graph::GetCOOCols() {
  CreateSparseFormat(_COO);
  return coo_->col;
}
torch::Tensor Graph::GetCOOEids() {
  CreateSparseFormat(_COO);
  return coo_->e_ids.has_value() ? coo_->e_ids.value() : torch::Tensor();
}
torch::Tensor Graph::GetCSRIndptr() {
  CreateSparseFormat(_CSR);
  return csr_->indptr;
}
torch::Tensor Graph::GetCSRIndices() {
  CreateSparseFormat(_CSR);
  return csr_->indices;
}
torch::Tensor Graph::GetCSREids() {
  CreateSparseFormat(_CSR);
  return csr_->e_ids.has_value() ? csr_->e_ids.value() : torch::Tensor();
}

// todo : not support compact
// axis == 0 for row, axis == 1 for column
std::tuple<c10::intrusive_ptr<Graph>, torch::Tensor> Graph::Slicing(
    torch::Tensor seeds, int64_t axis, int64_t on_format,
    int64_t output_format) {
  CreateSparseFormat(on_format);
  torch::Tensor select_index;
  std::shared_ptr<COO> coo_ptr = nullptr;
  std::shared_ptr<CSC> csc_ptr = nullptr;
  std::shared_ptr<CSR> csr_ptr = nullptr;
  std::shared_ptr<_TMP> tmp_ptr = nullptr;
  bool with_coo = output_format & _COO;
  int64_t new_num_cols, new_num_rows;
  torch::optional<torch::Tensor> e_ids = torch::nullopt;

  if (axis == 0) {
    new_num_cols = num_cols_;
    new_num_rows = seeds.numel();
  } else {
    new_num_cols = seeds.numel();
    new_num_rows = num_rows_;
  }
  auto ret = c10::intrusive_ptr<Graph>(
      std::unique_ptr<Graph>(new Graph(new_num_rows, new_num_cols)));

  if (on_format == _COO) {
    std::tie(coo_ptr, select_index) = COOSlicing(coo_, seeds, axis);
    e_ids = coo_ptr->e_ids;

  } else if (on_format == _CSC) {
    CHECK(output_format != _CSR)
        << "Error in Slicing, Not implementation [on_format = CSC, "
           "output_forat = CSR] !";
    e_ids = csc_->e_ids;

    if (axis == 0) {
      std::tie(tmp_ptr, select_index) = OnIndicesSlicing(csc_, seeds, with_coo);
    } else {
      std::tie(tmp_ptr, select_index) = OnIndptrSlicing(csc_, seeds, with_coo);
    }

    if (output_format & _CSC) {
      csc_ptr = std::make_shared<CSC>(
          CSC{tmp_ptr->indptr, tmp_ptr->coo_in_indices, torch::nullopt});
    }

    if (output_format & _COO) {
      coo_ptr = std::make_shared<COO>(COO{tmp_ptr->coo_in_indices,
                                          tmp_ptr->coo_in_indptr,
                                          torch::nullopt, false, true});
    }

  } else if (on_format == _CSR) {
    CHECK(output_format != _CSC)
        << "Error in Slicing, Not implementation [on_format = CSR, "
           "output_forat = CSC] !";
    e_ids = csr_->e_ids;

    if (axis == 0) {
      std::tie(tmp_ptr, select_index) = OnIndptrSlicing(csr_, seeds, with_coo);
    } else {
      std::tie(tmp_ptr, select_index) = OnIndicesSlicing(csr_, seeds, with_coo);
    }

    if (output_format & _CSR) {
      csr_ptr = std::make_shared<CSR>(
          CSR{tmp_ptr->indptr, tmp_ptr->coo_in_indices, torch::nullopt});
    }

    if (output_format & _COO) {
      coo_ptr = std::make_shared<COO>(COO{tmp_ptr->coo_in_indptr,
                                          tmp_ptr->coo_in_indices,
                                          torch::nullopt, true, false});
    }
  }

  ret->SetNumEdges(select_index.numel());
  ret->SetCOO(coo_ptr);
  ret->SetCSC(csc_ptr);
  ret->SetCSR(csr_ptr);

  torch::Tensor split_index;
  if (e_ids.has_value()) {
    split_index = e_ids.value().index_select(0, select_index);
  } else {
    split_index = select_index;
  }

  return {ret, split_index};
}

// axis == 0 for sample column and axis == 1 for sample row
std::tuple<c10::intrusive_ptr<Graph>, torch::Tensor> Graph::Sampling(
    int64_t axis, int64_t fanout, bool replace, int64_t on_format,
    int64_t output_format) {
  CreateSparseFormat(on_format);
  torch::Tensor select_index;
  std::shared_ptr<_TMP> tmp_ptr = nullptr;
  bool with_coo = output_format & _COO;
  torch::optional<torch::Tensor> e_ids = torch::nullopt;

  // sampling does not change the shape of graph/matrix
  auto ret = c10::intrusive_ptr<Graph>(
      std::unique_ptr<Graph>(new Graph(num_rows_, num_cols_)));

  if (axis == 0 && on_format == _CSC) {
    CHECK(output_format != _CSR)
        << "Error in Sampling, Not implementation [on_format = CSC, "
           "output_forat = CSR] !";

    std::tie(tmp_ptr, select_index) =
        CSCColSampling(csc_, fanout, replace, with_coo);

    if (output_format & _CSC)
      ret->SetCSC(std::make_shared<CSC>(
          CSC{tmp_ptr->indptr, tmp_ptr->coo_in_indices, torch::nullopt}));
    if (output_format & _COO)
      ret->SetCOO(std::make_shared<COO>(COO{tmp_ptr->coo_in_indices,
                                            tmp_ptr->coo_in_indptr,
                                            torch::nullopt, false, true}));

    e_ids = csc_->e_ids;
    ret->SetNumEdges(select_index.numel());
  } else if (axis == 1 && on_format == _CSR) {
    CHECK(output_format != _CSC)
        << "Error in Sampling, Not implementation [on_format = CSR, "
           "output_forat = CSC] !";

    std::tie(tmp_ptr, select_index) =
        CSCColSampling(csr_, fanout, replace, with_coo);

    if (output_format & _CSR)
      ret->SetCSR(std::make_shared<CSR>(
          CSR{tmp_ptr->indptr, tmp_ptr->coo_in_indices, torch::nullopt}));
    if (output_format & _COO)
      ret->SetCOO(std::make_shared<COO>(COO{tmp_ptr->coo_in_indptr,
                                            tmp_ptr->coo_in_indices,
                                            torch::nullopt, true, false}));

    e_ids = csr_->e_ids;
    ret->SetNumEdges(select_index.numel());
  } else {
    CHECK(false) << "Error in Sampling, Not implementation [axis = " << axis
                 << ", on_format = " << on_format
                 << ", output_forat = " << output_format << "] !";
  }

  torch::Tensor split_index;
  if (e_ids.has_value()) {
    split_index = e_ids.value().index_select(0, select_index);
  } else {
    split_index = select_index;
  }
  return {ret, split_index};
}

std::tuple<c10::intrusive_ptr<Graph>, torch::Tensor> Graph::SamplingProbs(
    int64_t axis, torch::Tensor edge_probs, int64_t fanout, bool replace,
    int64_t on_format, int64_t output_format) {
  CreateSparseFormat(on_format);
  torch::Tensor select_index;
  std::shared_ptr<_TMP> tmp_ptr = nullptr;
  bool with_coo = output_format & _COO;

  // sampling does not change the shape of graph/matrix
  auto ret = c10::intrusive_ptr<Graph>(
      std::unique_ptr<Graph>(new Graph(num_rows_, num_cols_)));

  if (axis == 0 && on_format == _CSC) {
    CHECK(output_format != _CSR)
        << "Error in SamplingProbs, Not implementation [on_format = CSC, "
           "output_forat = CSR] !";

    std::tie(tmp_ptr, select_index) =
        CSCColSamplingProbs(csc_, edge_probs, fanout, replace, with_coo);

    if (output_format & _CSC)
      ret->SetCSC(std::make_shared<CSC>(
          CSC{tmp_ptr->indptr, tmp_ptr->coo_in_indices, torch::nullopt}));
    if (output_format & _COO)
      ret->SetCOO(std::make_shared<COO>(COO{tmp_ptr->coo_in_indices,
                                            tmp_ptr->coo_in_indptr,
                                            torch::nullopt, false, true}));

    ret->SetNumEdges(select_index.numel());
  } else if (axis == 1 && on_format == _CSR) {
    CHECK(output_format != _CSC)
        << "Error in SamplingProbs, Not implementation [on_format = CSR, "
           "output_forat = CSC] !";

    std::tie(tmp_ptr, select_index) =
        CSCColSamplingProbs(csr_, edge_probs, fanout, replace, with_coo);

    if (output_format & _CSR)
      ret->SetCSR(std::make_shared<CSR>(
          CSR{tmp_ptr->indptr, tmp_ptr->coo_in_indices, torch::nullopt}));
    if (output_format & _COO)
      ret->SetCOO(std::make_shared<COO>(COO{tmp_ptr->coo_in_indptr,
                                            tmp_ptr->coo_in_indices,
                                            torch::nullopt, true, false}));

    ret->SetNumEdges(select_index.numel());
  } else {
    CHECK(false) << "Error in SamplingProbs, Not implementation [axis = "
                 << axis << ", on_format = " << on_format
                 << ", output_forat = " << output_format << "] !";
  }

  return {ret, select_index};
}

torch::Tensor Graph::RandomWalk(torch::Tensor seeds, int64_t walk_length) {
  return FusedRandomWalk(this->csc_, seeds, walk_length);
}
torch::Tensor Graph::Node2Vec(torch::Tensor seeds, int64_t walk_length,
                              double p, double q) {
  return FusedNode2Vec(this->csc_, seeds, walk_length, p, q);
}
}  // namespace gs
