
#include <gtest/gtest.h>
#include "hetero_graph.h"
#include "graph.h"
#include <torch/torch.h>

using namespace gs;

TEST(RandomWalk, testGraphWithOnePath)
{
    auto graph1 = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(new Graph(false)));
    auto graph2 = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(new Graph(false)));
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    int64_t indptr1[] = {0, 0, 1, 2, 3};
    int64_t indices1[] = {0, 1, 2};
    graph1->LoadCSC(torch::from_blob(indptr1, {5}, options).to(torch::kCUDA), torch::from_blob(indices1, {3}, options).to(torch::kCUDA));
    int64_t indptr2[] = {0, 1, 2, 3};
    int64_t indices2[] = {0, 1, 2};
    graph2->LoadCSC(torch::from_blob(indptr2, {4}, options).to(torch::kCUDA), torch::from_blob(indices2, {3}, options).to(torch::kCUDA));
    std::vector<std::string> node_types = {"user", "item"};
    std::tuple<std::string, std::string, std::string> edge1("user", "follow", "user");
    std::tuple<std::string, std::string, std::string> edge2("user", "view", "item");

    std::vector<std::tuple<std::string, std::string, std::string>>
        edge_types = {edge1, edge2};
    std::vector<c10::intrusive_ptr<Graph>> edge_relations = {graph1, graph2};
    HeteroGraph hg = HeteroGraph();
    hg.LoadFromHomo(node_types, edge_types, edge_relations);

    std::vector<int64_t> seeds_vector = {2, 1};
    std::vector<std::string> metapath = {"view", "follow", "follow"};
    torch::Tensor seeds = torch::from_blob(seeds_vector.data(), {2}, options).to(torch::kCUDA);
    auto actual_path = hg.MetapathRandomWalkFused(seeds, metapath);

    std::vector<int64_t> expected_path_vector = {2, 2, 1, 0, 1, 1, 0, -1};
    torch::Tensor expected_path = torch::from_blob(expected_path_vector.data(), {2, 4}, options).to(torch::kCUDA);
    std::cout << "expected path:" << expected_path << std::endl;
    std::cout << "actual path:" << actual_path << std::endl;
    EXPECT_TRUE(actual_path.equal(expected_path));
}

TEST(RandomWalk, testGraphWithMultiPath)
{  
    //homo graph1(user follow user): 0->1, 1->2, 2->3
    auto graph1 = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(new Graph(false)));
    auto graph2 = c10::intrusive_ptr<Graph>(std::unique_ptr<Graph>(new Graph(false)));
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    int64_t indptr1[] = {0, 0, 1, 2, 3};
    int64_t indices1[] = {0, 1, 2};
    graph1->LoadCSC(torch::from_blob(indptr1, {5}, options).to(torch::kCUDA), torch::from_blob(indices1, {3}, options).to(torch::kCUDA));
    //homo graph2(user view item): 0->0, 0->1, 1->1, 2->2, 3->2
    int64_t indptr2[] = {0, 1, 3, 5};
    int64_t indices2[] = {0, 0, 1, 2, 3};
    graph2->LoadCSC(torch::from_blob(indptr2, {4}, options).to(torch::kCUDA), torch::from_blob(indices2, {5}, options).to(torch::kCUDA));
    std::vector<std::string> node_types = {"user", "item"};
    std::tuple<std::string, std::string, std::string> edge1("user", "follow", "user");
    std::tuple<std::string, std::string, std::string> edge2("user", "view", "item");

    std::vector<std::tuple<std::string, std::string, std::string>>
        edge_types = {edge1, edge2};
    std::vector<c10::intrusive_ptr<Graph>> edge_relations = {graph1, graph2};
    HeteroGraph hg = HeteroGraph();
    hg.LoadFromHomo(node_types, edge_types, edge_relations);

    std::vector<int64_t> seeds_vector = {2, 2, 1, 1, 0};
    std::vector<std::string> metapath = {"view", "follow", "follow"};
    torch::Tensor seeds = torch::from_blob(seeds_vector.data(), {5}, options).to(torch::kCUDA);
    auto actual_path = hg.MetapathRandomWalkFused(seeds, metapath);
    /*
    if seed node is 2, there are two possible path: 2->2->1->0, 2->3->2->1, 
    if seed node is 1, there are two possible path: 1->0->-1->-1, 1->1->0->-1
    if seed node is 0, there are one possible path: 0->0->-1->-1
    */
    std::cout << "actual path:" << actual_path << std::endl;
}