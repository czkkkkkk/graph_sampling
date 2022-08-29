

#include "./hetero_graph.h"
#include "cuda/graph_ops.h"

namespace gs {

/**
 * @brief Build a HeteroGraph from a set of homogeneous graphs.
 *
 * @param node_types a set of node type and its size
 * @param edge_types a set of edge types with (src_type, edge_type, dest_type)
 * @param edge_ralations a set of homogeneous graphs describing the edges of
 * each edge type
 *
 */

// TODO: initialize n_nodes in NodeInfo
void HeteroGraph::LoadFromHomo(
    const std::vector<std::string>& node_types,
    const std::vector<std::tuple<std::string, std::string, std::string>>&
        edge_types,
    const std::vector<c10::intrusive_ptr<Graph>>& edge_relations) {
  this->n_node_types_ = node_types.size();
  this->n_edge_types_ = edge_types.size();
  // Note: currently nodeInfo set n_nodes to zero
  for (int64_t i = 0; i < this->n_node_types_; i++) {
    this->node_type_mapping_.insert(std::make_pair(node_types.at(i), i));
    NodeInfo nodeInfo = {i, 0, {}};
    this->hetero_nodes_.insert(std::make_pair(i, nodeInfo));
  }
  for (int64_t i = 0; i < this->n_edge_types_; i++) {
    edge_type_mapping_.insert(std::make_pair(std::get<1>(edge_types.at(i)), i));
    auto graph = edge_relations.at(i).get();
    EdgeRelation edge_relation = {
        this->node_type_mapping_[std::get<0>(edge_types.at(i))],
        this->node_type_mapping_[std::get<2>(edge_types.at(i))],
        this->edge_type_mapping_[std::get<1>(edge_types.at(i))],
        edge_relations.at(i),
        {}};
    hetero_edges_.insert(std::make_pair(i, edge_relation));
  }
}

c10::intrusive_ptr<Graph> HeteroGraph::GetHomoGraph(
    const std::string& edge_type) const {
  int64_t edge_id = edge_type_mapping_.at(edge_type);
  return hetero_edges_.at(edge_id).homo_graph;
}

torch::Tensor HeteroGraph::MetapathRandomWalk(
    torch::Tensor seeds, const std::vector<std::string>& metapath) {
  std::vector<torch::Tensor> ret;
  ret.push_back(seeds);
  for (std::string path : metapath) {
    c10::intrusive_ptr<Graph> homo_graph = this->GetHomoGraph(path);
    auto subA =
        homo_graph->ColumnwiseSlicing(seeds)->ColumnwiseSampling(1, true);
    torch::Tensor indptr = subA->GetCSC()->indptr;
    torch::Tensor indices = subA->GetCSC()->indices;
    seeds = subA->RowIndices();
    ret.push_back(seeds);
  }
  return torch::stack(ret);
}
torch::Tensor HeteroGraph::MetapathRandomWalkFused(
    torch::Tensor seeds, const std::vector<std::string>& metapath) {
  int64_t size = metapath.size();
  std::vector<int64_t> metapath_mapped(size);
  torch::Tensor homo_indptr_tensor, homo_indice_tensor;
  std::vector<int64_t> indptr_offset(this->n_edge_types_ + 1);
  std::vector<int64_t> indice_offset(this->n_edge_types_ + 1);
  indptr_offset[0] = 0;
  indice_offset[0] = 0;
  std::cout << __FILE__ << __LINE__ << std::endl;
  for (size_t i = 0; i < this->n_edge_types_; i++) {
    c10::intrusive_ptr<Graph> homograph = hetero_edges_.at(i).homo_graph;
    std::shared_ptr<CSC> csc = homograph->GetCSC();
    if (i == 0) {
      homo_indptr_tensor = csc->indptr.clone();
      homo_indice_tensor = csc->indices.clone();
    } else {
      homo_indptr_tensor = torch::cat({homo_indptr_tensor, csc->indptr});
      homo_indice_tensor = torch::cat({homo_indice_tensor, csc->indices});
    }
    indptr_offset[i + 1] = indptr_offset[i] + csc->indptr.numel();
    indice_offset[i + 1] = indice_offset[i] + csc->indices.numel();
  }
  std::cout << __FILE__ << __LINE__ << std::endl;

  for (size_t i = 0; i < size; i++) {
    metapath_mapped[i] = edge_type_mapping_[metapath[i]];
  }

  auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  std::cout << "options:" << opts << std::endl;
  torch::Tensor metapath_tensor =
      torch::from_blob(metapath_mapped.data(), size, opts).to(torch::kCUDA);
  torch::Tensor homo_indptr_offset =
      torch::from_blob(indptr_offset.data(), indptr_offset.size(), opts)
          .to(torch::kCUDA);
  torch::Tensor homo_indice_offset =
      torch::from_blob(indice_offset.data(), indice_offset.size(), opts)
          .to(torch::kCUDA);
  std::cout << __FILE__ << __LINE__ << std::endl;
  torch::Tensor paths = impl::MetapathRandomWalkFusedCUDA(
      seeds, metapath_tensor, homo_indptr_tensor, homo_indice_tensor,
      homo_indptr_offset, homo_indice_offset);
  return paths.reshape({seeds.numel(), -1});
}
}  // namespace gs