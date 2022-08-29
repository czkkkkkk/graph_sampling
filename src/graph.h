#ifndef GS_GRAPH_H_
#define GS_GRAPH_H_

#include "./logging.h"

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
  std::shared_ptr<CSC> GetCSC();
  c10::intrusive_ptr<Graph> ColumnwiseSlicing(torch::Tensor column_ids);
  c10::intrusive_ptr<Graph> ColumnwiseSampling(int64_t fanout, bool replace);
  c10::intrusive_ptr<Graph> ColumnwiseFusedSlicingAndSampling(
      torch::Tensor column_ids, int64_t fanout, bool replace);
  torch::Tensor RowIndices(bool unique);
  torch::Tensor AllIndices();
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Relabel();
  std::vector<torch::Tensor> MetaData();

  void Print() const;

 private:
  bool is_subgraph_;  // used for subgraph, stores node id in global graph
  std::shared_ptr<CSC> csc_;
};

}  // namespace gs

#endif
