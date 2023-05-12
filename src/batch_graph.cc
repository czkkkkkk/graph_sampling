#include "batch_graph.h"

#include <sstream>
#include "bcast.h"
#include "cuda/batch/batch_ops.h"
#include "cuda/fusion/column_row_slicing.h"
#include "cuda/fusion/edge_map_reduce.h"
#include "cuda/tensor_ops.h"
#include "graph_ops.h"

namespace gs {
// axis == 0 for row, axis == 1 for column
std::tuple<c10::intrusive_ptr<BatchGraph>, torch::Tensor>
BatchGraph::BatchColSlicing(torch::Tensor seeds, torch::Tensor batch_ptr,
                            int64_t axis, int64_t on_format,
                            int64_t output_format, bool encoding) {
  CreateSparseFormat(on_format);
  torch::Tensor select_index;
  std::shared_ptr<COO> coo_ptr = nullptr;
  std::shared_ptr<CSC> csc_ptr = nullptr;
  std::shared_ptr<CSR> csr_ptr = nullptr;
  std::shared_ptr<_TMP> tmp_ptr = nullptr;
  bool with_coo = output_format & _COO;
  int64_t new_num_cols, new_num_rows;
  torch::optional<torch::Tensor> e_ids = torch::nullopt;
  int64_t batch_num = batch_ptr.numel() - 1;
  col_bptr_ = batch_ptr;

  if (axis == 1) {
    new_num_cols = seeds.numel();
    new_num_rows = GetNumRows() * batch_num;
  } else {
    LOG(FATAL) << "batch slicing only suppurt column wise";
  }
  auto ret = c10::intrusive_ptr<BatchGraph>(
      std::unique_ptr<BatchGraph>(new BatchGraph(new_num_rows, new_num_cols)));

  if (on_format == _CSC) {
    CHECK(output_format != _CSR)
        << "Error in Slicing, Not implementation [on_format = CSC, "
           "output_forat = CSR] !";
    auto csc = GetCSC();
    e_ids = csc->e_ids;

    std::tie(tmp_ptr, select_index, edge_bptr_) = BatchOnIndptrSlicing(
        csc, seeds, batch_ptr, with_coo, encoding, GetNumRows());

    if (output_format & _CSC)
      csc_ptr = std::make_shared<CSC>(
          CSC{tmp_ptr->indptr, tmp_ptr->coo_in_indices, torch::nullopt});
    if (output_format & _COO) {
      coo_ptr = std::make_shared<COO>(COO{tmp_ptr->coo_in_indices,
                                          tmp_ptr->coo_in_indptr,
                                          torch::nullopt, false, true});
    }

  } else {
    LOG(FATAL) << "Not Implementatin Error";
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

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>,
           std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::string>
BatchGraph::GraphRelabel(torch::Tensor col_seeds, torch::Tensor row_ids) {
  auto csc = GetCSC();
  if (csc != nullptr) {
    if (col_bptr_.numel() == 0)
      LOG(FATAL)
          << "Relabel BatchGraph on CSC must has csc indptr batch pointer";
    if (edge_bptr_.numel() == 0) edge_bptr_ = csc->indptr.index({col_bptr_});

    torch::Tensor row_indices =
        row_ids.numel() > 0 ? row_ids.index({csc->indices}) : csc->indices;

    torch::Tensor unique_tensor, unique_tensor_bptr, out_indices, indices_bptr;

    std::tie(unique_tensor, unique_tensor_bptr, out_indices, indices_bptr) =
        impl::batch::BatchCSCRelabelCUDA(col_seeds, col_bptr_, row_indices,
                                         edge_bptr_);

    torch::Tensor relabeled_indptr = csc->indptr.clone();

    auto frontier_vector =
        impl::batch::SplitByOffset(unique_tensor, unique_tensor_bptr);
    auto indptr_vector =
        impl::batch::SplitIndptrByOffsetCUDA(csc->indptr, col_bptr_);
    auto indices_vector = impl::batch::SplitByOffset(out_indices, indices_bptr);
    std::vector<torch::Tensor> eid_vector;
    if (csc->e_ids.has_value())
      eid_vector = impl::batch::SplitByOffset(csc->e_ids.value(), indices_bptr);

    return {frontier_vector, indptr_vector, indices_vector, eid_vector, "csc"};
  } else {
    if (edge_bptr_.numel() == 0)
      LOG(FATAL) << "Relabel BatchGraph on COO must has edge batch pointer";

    CreateSparseFormat(_COO);
    auto coo = GetCOO();

    if (!coo->col_sorted)
      LOG(FATAL)
          << "Relabel BatchGraph on COO must require COO to be column-sorted";

    torch::Tensor coo_col = coo->col;
    torch::Tensor coo_row =
        row_ids.numel() > 0 ? row_ids.index({coo->row}) : coo->row;

    torch::Tensor unique_tensor, unique_tensor_bptr;
    torch::Tensor out_coo_row, out_coo_col, out_coo_bptr;
    std::tie(unique_tensor, unique_tensor_bptr, out_coo_row, out_coo_col,
             out_coo_bptr) =
        impl::batch::BatchCOORelabelCUDA(col_seeds, col_bptr_, coo_col, coo_row,
                                         edge_bptr_);

    auto frontier_vector =
        impl::batch::SplitByOffset(unique_tensor, unique_tensor_bptr);
    auto coo_row_vector = impl::batch::SplitByOffset(out_coo_row, out_coo_bptr);
    auto coo_col_vector =
        impl::batch::SplitIndptrByOffsetCUDA(out_coo_col, out_coo_bptr);
    std::vector<torch::Tensor> eid_vector;
    if (csc->e_ids.has_value())
      eid_vector = impl::batch::SplitByOffset(coo->e_ids.value(), out_coo_bptr);

    return {frontier_vector, coo_row_vector, coo_col_vector, eid_vector, "coo"};
  }
}
}  // namespace gs