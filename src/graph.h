#ifndef GS_GRAPH_H_
#define GS_GRAPH_H_

#include <torch/custom_class.h>
#include <torch/script.h>

#include "./graph_storage.h"

namespace gs {

class Graph : public torch::CustomClassHolder {
 public:
  Graph(bool is_subgraph) { is_subgraph_ = is_subgraph; }
  Graph(bool is_subgraph, torch::optional<torch::Tensor> col_ids,
        torch::optional<torch::Tensor> row_ids, int64_t num_cols,
        int64_t num_rows)
      : is_subgraph_{is_subgraph},
        num_cols_{num_cols},
        num_rows_{num_rows},
        col_ids_{col_ids},
        row_ids_{row_ids} {}
  void LoadCSC(torch::Tensor indptr, torch::Tensor indices);
  void LoadCSR(torch::Tensor indptr, torch::Tensor indices);
  void LoadCSCWithColIds(torch::Tensor column_ids, torch::Tensor indptr,
                         torch::Tensor indices);
  void SetCSC(std::shared_ptr<CSC> csc);
  void SetCSR(std::shared_ptr<CSR> csr);
  void SetCOO(std::shared_ptr<COO> coo);
  void SetData(torch::Tensor data);
  void SetValidCols(torch::Tensor val_cols);
  void SetValidRows(torch::Tensor val_rows);
  void SetNumEdges(int64_t num_edges);
  void CSC2CSR();
  void CSC2DCSR();
  void CSR2CSC();
  void CSR2DCSC();
  void CreateCOO();
  void CreateCSR();
  void CreateCSC();
  std::shared_ptr<CSC> GetCSC();
  std::shared_ptr<CSR> GetCSR();
  std::shared_ptr<COO> GetCOO();
  torch::optional<torch::Tensor> GetData();
  int64_t GetNumRows();
  int64_t GetNumCols();
  int64_t GetNumEdges();
  c10::intrusive_ptr<Graph> FusedBidirSlicing(torch::Tensor column_seeds,
                                              torch::Tensor row_seeds);
  c10::intrusive_ptr<Graph> ColumnwiseSlicing(torch::Tensor column_index);
  c10::intrusive_ptr<Graph> RowwiseSlicing(torch::Tensor row_index);
  c10::intrusive_ptr<Graph> ColumnwiseSampling(int64_t fanout, bool replace);
  c10::intrusive_ptr<Graph> ColumnwiseSamplingProbs(torch::Tensor edge_probs,
                                                    int64_t fanout,
                                                    bool replace);
  c10::intrusive_ptr<Graph> ColumnwiseFusedSlicingAndSampling(
      torch::Tensor column_index, int64_t fanout, bool replace);
  torch::Tensor Sum(int64_t axis, int64_t powk, int64_t on_format);
  c10::intrusive_ptr<Graph> Divide(torch::Tensor divisor, int64_t axis);
  c10::intrusive_ptr<Graph> Normalize(int64_t axis);
  // A "valid" node means that the node is required by the user or that it is
  // not an isolated node.
  torch::Tensor AllValidNode();
  torch::Tensor GetRows();       // return row_ids
  torch::Tensor GetCols();       // return col_ids
  torch::Tensor GetValidRows();  // return valid row_ids.
  torch::Tensor GetValidCols();  // return valid col_ids.

  //  If is_original, it return  in global_id else in local_id.
  torch::Tensor GetCOORows(bool is_original);  // return coo_row, which is
                                               // coo[0]
  torch::Tensor GetCOOCols(bool is_original);  // return coo_row, which is
                                               // coo[1]
  std::tuple<torch::Tensor, int64_t, int64_t, torch::Tensor, torch::Tensor,
             torch::optional<torch::Tensor>, std::string>
  Relabel();
  std::vector<torch::Tensor> MetaData();
  torch::Tensor RandomWalk(torch::Tensor seeds, int64_t walk_length);
  void SDDMM(const std::string& op, torch::Tensor lhs, torch::Tensor rhs,
             torch::Tensor out, int64_t lhs_target, int64_t rhs_target);

  // todo: return global_e_id
 private:
  bool is_subgraph_;
  int64_t num_cols_;  // total number of cols in a matrix
  int64_t num_rows_;  // total number of rows in a matrix
  int64_t num_edges_;
  std::shared_ptr<CSC> csc_;
  std::shared_ptr<CSR> csr_;
  std::shared_ptr<COO> coo_;
  torch::optional<torch::Tensor> data_;
  torch::optional<torch::Tensor> col_ids_;      // column id in matrix
  torch::optional<torch::Tensor> row_ids_;      // row id in matrix
  torch::optional<torch::Tensor> val_col_ids_;  // valid column id in matrix
  torch::optional<torch::Tensor> val_row_ids_;  // valid row id in matrix

  void CreateSparseFormat(int64_t axis);
};

}  // namespace gs

#endif
