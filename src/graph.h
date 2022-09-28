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
  void CSC2CSR();
  void CSR2CSC();
  std::shared_ptr<CSC> GetCSC();
  std::shared_ptr<CSR> GetCSR();
  std::shared_ptr<COO> GetCOO();
  torch::optional<torch::Tensor> GetData();
  int64_t GetNumRows();
  int64_t GetNumCols();
  c10::intrusive_ptr<Graph> ColumnwiseSlicing(torch::Tensor column_index);
  c10::intrusive_ptr<Graph> RowwiseSlicing(torch::Tensor row_index);
  c10::intrusive_ptr<Graph> ColumnwiseSampling(int64_t fanout, bool replace);
  c10::intrusive_ptr<Graph> ColumnwiseFusedSlicingAndSampling(
      torch::Tensor column_index, int64_t fanout, bool replace);
  torch::Tensor Sum(int64_t axis);
  torch::Tensor L2Norm(int64_t axis);
  c10::intrusive_ptr<Graph> Divide(torch::Tensor divisor, int64_t axis);
  c10::intrusive_ptr<Graph> Normalize(int64_t axis);
  torch::Tensor RowIndices(bool unique);
  torch::Tensor AllIndices(bool unique);
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
             torch::optional<torch::Tensor>, std::string>
  Relabel();
  std::vector<torch::Tensor> MetaData();

  void Print() const;

 private:
  bool is_subgraph_;
  int64_t num_cols_;  // total number of cols in a matrix
  int64_t num_rows_;  // total number of rows in a matrix
  std::shared_ptr<CSC> csc_;
  std::shared_ptr<CSR> csr_;
  std::shared_ptr<COO> coo_;
  torch::optional<torch::Tensor> data_;
  torch::optional<torch::Tensor> col_ids_;  // column id in matrix
  torch::optional<torch::Tensor> row_ids_;  // row id in matrix

  std::tuple<torch::Tensor, torch::optional<torch::Tensor>,
             torch::optional<torch::Tensor>>
  PrepareDataForCompute(int64_t axis);
};

}  // namespace gs

#endif
