#include <gtest/gtest.h>
#include "graph.h"
#include <torch/torch.h>

using namespace gs;

TEST(RowwiseSlicing, test1)
{
    Graph A(false);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    torch::Tensor indptr = torch::arange(0, 21, options).to(torch::kCUDA) * 5;
    torch::Tensor indices = torch::arange(0, 100, options).to(torch::kCUDA);
    A.LoadCSC(indptr, indices);

    torch::Tensor row_ids = torch::arange(0, 100, 5, options).to(torch::kCUDA);
    auto subA = A.RowwiseSlicing(row_ids);
    auto csc_ptr = subA->GetCSC();
    EXPECT_TRUE(csc_ptr->indptr.equal(torch::arange(21).to(torch::kCUDA) * 1));
    EXPECT_TRUE(csc_ptr->indices.equal(row_ids));
}