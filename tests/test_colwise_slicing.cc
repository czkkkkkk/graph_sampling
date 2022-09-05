#include <gtest/gtest.h>
#include "graph.h"
#include <torch/torch.h>

using namespace gs;

TEST(ColwiseSlicing, test1)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
    torch::Tensor indptr = torch::arange(0, 21, options) * 5;
    torch::Tensor indices = torch::arange(0, 100, options);
    A.LoadCSC(indptr, indices);

    torch::Tensor col_ids = torch::arange(0, 3, options);
    auto subA = A.ColumnwiseSlicing(col_ids);
    auto csc_ptr = subA->GetCSC();
    EXPECT_TRUE(csc_ptr->indptr.equal(torch::arange(0, 4, options) * 5));
    EXPECT_TRUE(csc_ptr->indices.equal(torch::arange(0, 15, options)));
}