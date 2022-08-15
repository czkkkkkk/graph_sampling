#include <gtest/gtest.h>
#include "graph.h"
#include <torch/torch.h>

using namespace gs;

TEST(Slicing, test1)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    int64_t indptr[] = {0, 1, 1, 3, 4};
    int64_t indices[] = {4, 0, 1, 2};
    int64_t res_indptr[] = {0, 2, 3};
    int64_t res_indices[] = {0, 1, 2};
    A.LoadCSC(torch::from_blob(indptr, {5}, options), torch::from_blob(indices, {4}, options));
    int64_t column_ids[] = {2, 3};
    auto subA = A.ColumnwiseSlicing(torch::from_blob(column_ids, {2}, options));
    auto csc_ptr = subA->GetCSC();
    EXPECT_TRUE(csc_ptr->indptr.equal(torch::from_blob(res_indptr, {3}, options)));
    EXPECT_TRUE(csc_ptr->indices.equal(torch::from_blob(res_indices, {3}, options)));
}