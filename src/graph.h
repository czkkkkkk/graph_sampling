#ifndef GS_GRAPH_H_
#define GS_GRAPH_H_

#include <torch/custom_class.h>
#include <torch/script.h>

#include "./graph_storage.h"

namespace gs {

class Graph : public torch::CustomClassHolder {
 public:
  Graph(bool is_subgraph) { is_subgraph_ = is_subgraph; }
  void LoadCSC(torch::Tensor indptr, torch::Tensor indices);
  void LoadCSCWithColIds(torch::Tensor column_ids, torch::Tensor indptr,
                         torch::Tensor indices);
  void SetCSC(std::shared_ptr<CSC> csc);
  void SetCSR(std::shared_ptr<CSR> csr);
  void SetCOO(std::shared_ptr<COO> coo);
  void CSC2CSR();
  std::shared_ptr<CSC> GetCSC();
  std::shared_ptr<CSR> GetCSR();
  std::shared_ptr<COO> GetCOO();
  c10::intrusive_ptr<Graph> ColumnwiseSlicing(torch::Tensor column_ids);
  c10::intrusive_ptr<Graph> RowwiseSlicing(torch::Tensor row_ids);
  c10::intrusive_ptr<Graph> ColumnwiseSampling(int64_t fanout, bool replace);
  c10::intrusive_ptr<Graph> ColumnwiseFusedSlicingAndSampling(
      torch::Tensor column_ids, int64_t fanout, bool replace);
  c10::intrusive_ptr<Graph> Normalize(int axis);
  torch::Tensor RowIndices();
  torch::Tensor AllIndices();
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Relabel();
  std::vector<torch::Tensor> MetaData();

  void Print() const;

 private:
  bool is_subgraph_;  // used for subgraph, stores node id in global graph
  std::shared_ptr<CSC> csc_;
  std::shared_ptr<CSR> csr_;
  std::shared_ptr<COO> coo_;
};

}  // namespace gs

#endif
